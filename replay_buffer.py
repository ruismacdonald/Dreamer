import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """
    Replay buffer supporting both FIFO and LoFo sampling.

    In FIFO mode (distance_process=False), sequences are sampled uniformly at random from all stored transitions (standard experience replay strategy).

    In LoFo mode, each transition stores a low-dimensional state-distance representation produced by a pretrained contrastive model. On every add(), distances from the new transition's representation to all stored transitions are computed. If the number of transitions within D_local (obs_repr_rad) exceeds N_local (obs_repr_count), the oldest transition in the neighbourhood has its kept mask value set to 0.0 (local forgetting). Sequences are sampled uniformly at random with the constraint that the start index must be kept (kept == 1.0). Discarded transitions within a sequence are still used in the RSSM forward pass to maintain temporal continuity, but their loss terms are zeroed out so they contribute no gradient signal.

    Parameters
        size: int – maximum number of transitions to store
        obs_shape: tuple – shape of a single observation (C, H, W)
        action_size: int – dimensionality of the action space
        seq_len: int – number of timesteps per sampled sequence
        batch_size: int – number of sequences per batch
        distance_process: bool – enable LoFo mode (default False -> pure FIFO)
        obs_repr_rad: float – D_local: neighbourhood radius in representation space
        obs_repr_count: int – N_local: max number of transitions allowed in the local 
                        neighbourhood before the oldest is discarded
        obs_repr_size: int – dimensionality of the state-distance representation vector; required 
                        when distance_process=True
    """

    def __init__(
        self,
        size,
        obs_shape,
        action_size,
        seq_len,
        batch_size,
        distance_process=False,
        obs_repr_rad=0.05,
        obs_repr_count=10,
        obs_repr_size=32,
    ):
        self.size = size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0  # Next write position (oldest data when buffer is full)
        self.full = False  # True once every slot has been written at least once
        self.steps = 0
        self.episodes = 0

        # Core transition arrays
        self.observations = np.empty((size, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.terminals = np.empty((size,), dtype=np.float32)

        # LoFo bookkeeping
        self.distance_process = distance_process
        self.obs_repr_rad = obs_repr_rad
        self.obs_repr_count = obs_repr_count

        if self.distance_process:
            assert obs_repr_size is not None, \
                "obs_repr_size must be provided when distance_process=True"
            self.obs_repr_size = obs_repr_size
            # Contiguous float32 array; all zeros until a slot is first written
            self.representations = np.zeros((size, obs_repr_size), dtype=np.float32)
            self.kept = np.ones((size,), dtype=np.float32)  # 1.0 = kept, 0.0 = discarded
            # LoFo sampling is only activated once every buffer slot has a real representation (i.e. after the first full cycle of the buffer). Before that we fall back to pure FIFO.
            self.repr_filled = False

    # Adding transitions to buffer

    def add(self, obs, ac, rew, done, representation=None):
        """
        Add a single transition to the buffer and update the kept mask.

        In LoFo mode, after writing the new transition, computes distances from its representation to all currently stored transitions. If the number of transitions within D_local (the local neighbourhood) exceeds N_local, the oldest transition in the neighbourhood has its kept mask value set to 0.0 (discarded). All newly written transitions start as kept (1.0); a slot being overwritten is reset to 1.0 before the neighbourhood check.

        Parameters
            obs: dict with key 'image' (np.ndarray uint8, shape obs_shape)
            ac: np.ndarray, shape (action_size,)
            rew: float
            done: bool or float — episode termination flag
            representation: np.ndarray of shape (obs_repr_size,), or None State-distance embedding 
                            from SimpleContrastiveStateDistanceModel. Must be provided on every call when distance_process=True.
        """
        self.observations[self.idx] = obs['image']
        self.actions[self.idx] = ac
        self.rewards[self.idx] = rew
        self.terminals[self.idx] = done

        if self.distance_process:
            self.kept[self.idx] = 1.0
            if representation is not None:
                if isinstance(representation, torch.Tensor):
                    representation = representation.detach().cpu().numpy()
                self.representations[self.idx] = representation
                
                # Compute distances from new transition to all valid stored transitions
                buf = self._valid_buffer_size()  # Call before incrementing idx
                reprs = self.representations[:buf]  # (buf, repr_size)
                # L2 (Euclidean) distance: sqrt(sum of squared differences) per stored transition
                dists = np.linalg.norm(reprs - representation, axis=-1)  # (buf,)
                
                if self.steps < 100000 and self.steps % 10000 == 0 and buf > 0:
                    print(f"[LoFo diag] step={self.steps} dist_mean={dists.mean():.4f} "
                        f"p10={np.percentile(dists,10):.4f} p20={np.percentile(dists,20):.4f} "
                        f"p50={np.percentile(dists,50):.4f}")
                    
                # Find all transitions in the local neighbourhood
                neighbours = np.where(dists <= self.obs_repr_rad)[0]  # Indices within D_local
                
                # If neighbourhood exceeds N_local, discard the oldest (lowest buffer index)
                if len(neighbours) > self.obs_repr_count:
                    oldest = neighbours[np.argmin(neighbours)]  # Lowest index = oldest
                    self.kept[oldest] = 0.0

        self.full = self.full or (self.idx + 1 == self.size)
        self.idx = (self.idx + 1) % self.size
        self.steps += 1
        self.episodes += int(done)

    # Index helpers (shared between FIFO and LoFoV1)

    def _valid_buffer_size(self):
        """Number of transitions currently stored (<= size)."""
        return self.size if self.full else self.idx

    def _sample_idx(self, L, require_kept=False):
        """
        Picks a random start position and returns L consecutive buffer indices (one sequence, as raw buffer indices).

        The buffer is circular so indices wrap modulo size. A block is rejected if it would cross the write head (self.idx), which marks the boundary between the newest and oldest data. Crossing that boundary would produce a sequence that is not contiguous in time — mixing recent data on one side with old overwritten data on the other — corrupting the RSSM's temporal rollout. We check idxs[1:] (not idxs[0]) because a sequence that starts at the write head is fine; it's only dangerous if the write head appears partway through or at the end of the sequence.

        Parameters
            L: int – sequence length
            require_kept: bool – if True (LoFo mode), also reject start positions whose kept value 
                            is 0.0 (discarded). Steps after the start may still be discarded; their precomputed kept values are retrieved in _retrieve_batch.
        """
        buf = self._valid_buffer_size()
        valid = False
        while not valid:
            start = np.random.randint(0, buf - L + 1)
            idxs = np.arange(start, start + L) % self.size
            if self.idx in idxs[1:]:
                continue  # Straddles write head
            if require_kept and self.kept[start] == 0.0:
                continue  # Start is discarded
            valid = True
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        """
        Gather transition data for a batch of sequences.

        Parameters
            idxs: np.ndarray, shape (n, L) – buffer indices for each sequence
            n: int – batch size
            L: int – sequence length

        Returns
            obs, acs, rews, terms — each shaped (L, n, ...) as expected by the
                                    RSSM which processes time as the leading dimension.
        """
        vec = idxs.transpose().reshape(-1)  # (L*n,) — unroll for fancy indexing
        obs = self.observations[vec].reshape(L, n, *self.obs_shape)
        acs = self.actions[vec].reshape(L, n, -1)
        rews = self.rewards[vec].reshape(L, n)
        terms = self.terminals[vec].reshape(L, n)
        kept = self.kept[vec].reshape(L, n) if self.distance_process else None
        return obs, acs, rews, terms, kept

    # LoFoV1 helpers
   
    def _representations_valid(self):
        """
        True when LoFo sampling can be used.

        LoFo requires every buffer slot to hold a real representation. Before the buffer has completed its first full cycle, some slots still contain the zero-initialised default, which would produce meaningless distances in add() and corrupt the kept mask. Once self.full flips True for the first time, repr_filled is latched permanently to True.
        """
        if not self.distance_process:
            return False
        if not self.repr_filled:
            self.repr_filled = self.steps > 10000
        return self.repr_filled

    # Sampling

    def sample(self):
        """
        Samples self.batch_size sequences of length self.seq_len, retrieves transition data (obs, actions, rewards, terminals) and the precomputed per-timestep kept mask for those indices.

        The kept mask (shape (L, n)) is stored per-transition in self.kept and updated in add() — it is not computed here. Each value is 1.0 (use this timestep's loss) or 0.0 (discard: zero out loss, no gradient signal).

        Note: per-timestep is equivalent to per-index even when the buffer is circular. Each index always maps to exactly one valid timestep once written. Circularity only invalidates certain contiguous blocks of indices — specifically those that wrap around past the write head (self.idx), which would mix transitions from the newest and oldest parts of the buffer into a single sequence that is not contiguous in time.

        In FIFO mode, sequences are drawn uniformly at random and all steps are marked kept (mask all ones).

        In LoFo mode, sequences are drawn uniformly at random with the additional constraint that the start index must not be discarded (kept == 1.0). Steps after the start may still have kept == 0.0 per their precomputed values; they are included for RSSM temporal continuity but contribute no gradient signal.

        Returns
            obs: np.ndarray, shape (L, n, *obs_shape), uint8
            acs: np.ndarray, shape (L, n, action_size), float32
            rews: np.ndarray, shape (L, n), float32
            terms: np.ndarray, shape (L, n), float32
            kept: np.ndarray, shape (L, n), float32 — 1.0 = use step, 0.0 = discard
        """
        n = self.batch_size
        L = self.seq_len
        lofo = self.distance_process and self._representations_valid()

        chosen = [self._sample_idx(L, require_kept=lofo) for _ in range(n)]
        idxs = np.asarray(chosen)  # (n, L)
        obs, acs, rews, terms, kept = self._retrieve_batch(idxs, n, L)

        if not lofo:
            kept = np.ones((L, n), dtype=np.float32)

        return obs, acs, rews, terms, kept

    # Utils

    def get_data(self):
        """
        Return all currently stored transitions as a dict of arrays. Used for offline training of the state-distance model before the main Dreamer training loop begins.
        """
        buf = self._valid_buffer_size()
        return {
            'observations': self.observations[:buf],
            'actions': self.actions[:buf],
            'rewards': self.rewards[:buf],
            'terminals': self.terminals[:buf],
        }

    def report_statistics(self):
        """Return buffer occupancy statistics for logging."""
        buf = self._valid_buffer_size()
        return {
            'buffer_size': buf,
            'buffer_steps': self.steps,
            'buffer_episodes': self.episodes,
            'kept_fraction': float(self.kept[:buf].mean()) if self.distance_process else 1.0,
        }

    def save(self, dname, fname='replay_buffer.pkl'):
        import pickle, os
        os.makedirs(dname, exist_ok=True)
        payload = {
            'idx': self.idx, 'full': self.full,
            'steps': self.steps, 'episodes': self.episodes,
            'observations': self.observations, 'actions': self.actions,
            'rewards': self.rewards, 'terminals': self.terminals,
        }
        if self.distance_process:
            payload.update({
                'representations': self.representations,
                'kept': self.kept,
                'repr_filled': self.repr_filled,
            })
        with open(os.path.join(dname, fname), 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, dname, fname='replay_buffer.pkl'):
        import pickle, os
        with open(os.path.join(dname, fname), 'rb') as f:
            p = pickle.load(f)
        self.idx = p['idx']; self.full = p['full']
        self.steps = p['steps']; self.episodes = p['episodes']
        self.observations = p['observations']; self.actions = p['actions']
        self.rewards = p['rewards']; self.terminals = p['terminals']
        if self.distance_process:
            self.representations = p['representations']
            self.kept = p['kept']
            self.repr_filled = p['repr_filled']


class ReplayBufferLoFoV2:
    """
    LoFoV2 replay buffer: SimHash-indexed per-region FIFO buffers.

    Uses the same global ring buffer as ReplayBuffer for transition storage so that Dreamer's 
    temporal sequence sampling works identically. Per-hash FIFOs store only ring indices (plus a 
    generation stamp to detect stale references after a slot is overwritten). The kept mask 
    controls which ring positions are eligible as sequence start indices.

    Normalization of representations before hashing is handled by 
    SimpleContrastiveStateDistanceModel.get_representation() when normalize_representations=True, 
    so this class receives already-normalized embeddings and only needs to apply the random 
    projection.

    Parameters
        size: int – global ring buffer capacity
        obs_shape: tuple – observation shape (C, H, W)
        action_size: int – action dimensionality
        seq_len: int – sequence length for sampling
        batch_size: int – sequences per batch
        obs_repr_size: int – representation dimensionality d
        obs_hash_count: int – per-region FIFO capacity N_local
        obs_hash_size: int – SimHash dimension h (number of hyperplanes)
        seed: int – RNG seed for the projection matrix
    """

    def __init__(
        self,
        size,
        obs_shape,
        action_size,
        seq_len,
        batch_size,
        obs_repr_size=32,
        obs_hash_count=2000,
        obs_hash_size=32,
        seed=0,
    ):
        self.size = size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.steps = 0
        self.episodes = 0

        # Global ring buffer
        self.idx  = 0
        self.full = False
        self.observations = np.empty((size, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.terminals = np.empty((size,), dtype=np.float32)
        self.kept = np.zeros((size,), dtype=np.float32)

        # Per-slot generation stamps (detect stale FIFO references after overwrite)
        self.insert_id = np.zeros(size, dtype=np.int64)
        self._global_insert_id = 0

        # Flat set of currently-kept start indices (O(1) add/remove via swap-delete)
        self.kept_flat = []  # list of ring indices
        self.flat_pos = np.full(size, -1, dtype=np.int64)  # ring idx -> position in kept_flat

        # SimHash
        self.obs_hash_count = obs_hash_count
        self.hash_bits = obs_hash_size
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal(size=(obs_hash_size, obs_repr_size)).astype(np.float32)

        # Per-hash FIFOs: bytes key -> deque of (ring_idx, insert_id)
        self.hash_fifos: dict[bytes, deque] = {}

        self.phase_labels = np.zeros(size, dtype=np.uint8)  # 1=phase1, 2=phase2


    # Flat kept-set helpers

    def _flat_add(self, idx):
        if self.flat_pos[idx] != -1:
            return
        self.flat_pos[idx] = len(self.kept_flat)
        self.kept_flat.append(idx)

    def _flat_remove(self, idx):
        pos = int(self.flat_pos[idx])
        if pos == -1:
            return
        last = self.kept_flat[-1]
        self.kept_flat[pos] = last
        self.flat_pos[last] = pos
        self.kept_flat.pop()
        self.flat_pos[idx] = -1


    # SimHash

    def _hash_key(self, rep: np.ndarray) -> bytes:
        """Map a (d,) representation to a bytes key via SimHash."""
        rep = np.asarray(rep, dtype=np.float32).reshape(-1)
        bits = (self.A @ rep >= 0).astype(np.uint8)
        return np.packbits(bits, bitorder='little').tobytes()


    # Writing

    def add(self, obs, ac, rew, done, representation, phase=1):
        """
        Write one transition and update the per-hash FIFO.

        Parameters
            obs: dict with key 'image'
            ac: np.ndarray, shape (action_size,)
            rew: float
            done: bool or float
            representation: np.ndarray, shape (obs_repr_size,) — required
            phase: int, 1=phase1, 2=phase2
        """
        assert representation is not None, \
            "ReplayBufferLoFoV2 requires a representation on every add()"

        i = self.idx

        # Write transition
        self.observations[i] = obs['image']
        self.actions[i] = ac
        self.rewards[i] = rew
        self.terminals[i] = done

        # Overwriting an old slot: remove it from kept set
        if self.kept[i] == 1.0:
            self._flat_remove(i)
        self.kept[i] = 1.0

        # Stamp this slot so old FIFO references become stale
        self._global_insert_id += 1
        self.insert_id[i] = self._global_insert_id

        # Add to per-hash FIFO and evict if over capacity
        if isinstance(representation, torch.Tensor):
            representation = representation.detach().cpu().numpy()
        key = self._hash_key(representation)
        fifo = self.hash_fifos.setdefault(key, deque())
        self._flat_add(i)
        fifo.append((i, self.insert_id[i]))

        while len(fifo) > self.obs_hash_count:
            disc_idx, disc_stamp = fifo.popleft()
            # Skip stale references (slot was overwritten since bucketing)
            if self.insert_id[disc_idx] != disc_stamp:
                continue
            # Live eviction
            self.kept[disc_idx] = 0.0
            self._flat_remove(disc_idx)
            break

        self.full = self.full or (self.idx + 1 == self.size)
        self.idx = (self.idx + 1) % self.size
        self.steps += 1
        self.episodes += int(done)

        self.phase_labels[i] = phase


    # Sampling

    def _sample_idx(self, L):
        """
        Sample one start from kept_flat, return L consecutive ring indices.
        Rejects sequences that straddle the write head.
        """
        while True:
            start = int(np.random.choice(self.kept_flat))
            idxs = np.arange(start, start + L) % self.size
            if self.idx not in idxs[1:]:
                return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec = idxs.transpose().reshape(-1)
        obs = self.observations[vec].reshape(L, n, *self.obs_shape)
        acs = self.actions[vec].reshape(L, n, -1)
        rews = self.rewards[vec].reshape(L, n)
        terms = self.terminals[vec].reshape(L, n)
        kept = self.kept[vec].reshape(L, n)
        return obs, acs, rews, terms, kept

    def sample(self):
        """
        Returns (obs, acs, rews, terms, kept), each (L, n, ...).
        kept reflects which steps were not locally forgotten.
        """
        assert len(self.kept_flat) > 0, \
            "No kept indices available yet — collect more data before sampling."
        n = self.batch_size
        L = self.seq_len
        idxs = np.asarray([self._sample_idx(L) for _ in range(n)])
        return self._retrieve_batch(idxs, n, L)


    # Utilities

    def _valid_buffer_size(self):
        return self.size if self.full else self.idx

    def get_data(self):
        """All stored transitions; used to train the state-distance model."""
        buf = self._valid_buffer_size()
        return {
            'observations': self.observations[:buf],
            'terminals':   self.terminals[:buf],
        }

    def report_statistics(self):
        fifo_sizes = [len(dq) for dq in self.hash_fifos.values()]
        n_full = sum(1 for s in fifo_sizes if s >= self.obs_hash_count)
        n_evicting = sum(1 for s in fifo_sizes if s > self.obs_hash_count)

        buf = self._valid_buffer_size()
        phase2_mask = self.phase_labels[:buf] == 2
        phase1_kept = float(self.kept[:buf][~phase2_mask].mean()) if (~phase2_mask).any() else 1.0
        phase2_kept = float(self.kept[:buf][phase2_mask].mean()) if phase2_mask.any() else 1.0

        return {
            'buffer_size': self._valid_buffer_size(),
            'buffer_steps': self.steps,
            'buffer_episodes': self.episodes,
            'buffer_n_regions': len(self.hash_fifos),
            'kept_starts': len(self.kept_flat),
            'fifo_size_mean': float(np.mean(fifo_sizes)) if fifo_sizes else 0.0,
            'fifo_size_max': int(np.max(fifo_sizes))   if fifo_sizes else 0,
            'fifo_n_full': n_full,  # How many regions have hit the cap
            'fifo_n_evicting': n_evicting,  # How many regions are evicting
            'kept_fraction': len(self.kept_flat) / self._valid_buffer_size() if self._valid_buffer_size() > 0 else 1.0,
            'phase1_kept_fraction': phase1_kept,
            'phase2_kept_fraction': phase2_kept,
        }
    
    def save(self, dname, fname='replay_buffer.pkl'):
        import pickle, os
        os.makedirs(dname, exist_ok=True)
        payload = {
            'idx': self.idx, 'full': self.full,
            'steps': self.steps, 'episodes': self.episodes,
            'observations': self.observations, 'actions': self.actions,
            'rewards': self.rewards, 'terminals': self.terminals,
            'kept': self.kept, 'insert_id': self.insert_id,
            '_global_insert_id': self._global_insert_id,
            'kept_flat': self.kept_flat, 'flat_pos': self.flat_pos,
            'hash_fifos': self.hash_fifos,
            'phase_labels': self.phase_labels,
            'A': self.A,
        }
        with open(os.path.join(dname, fname), 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, dname, fname='replay_buffer.pkl'):
        import pickle, os
        with open(os.path.join(dname, fname), 'rb') as f:
            p = pickle.load(f)
        self.idx = p['idx']; self.full = p['full']
        self.steps = p['steps']; self.episodes = p['episodes']
        self.observations = p['observations']; self.actions = p['actions']
        self.rewards = p['rewards']; self.terminals = p['terminals']
        self.kept = p['kept']; self.insert_id = p['insert_id']
        self._global_insert_id = p['_global_insert_id']
        self.kept_flat = p['kept_flat']; self.flat_pos = p['flat_pos']
        self.hash_fifos = p['hash_fifos']
        self.phase_labels = p['phase_labels']
        self.A = p['A']