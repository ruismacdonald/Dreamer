import argparse
import collections
import functools
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import re
import subprocess
import warnings

os.environ['NO_AT_BRIDGE'] = '1'  # Hide X org false warning.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '20'
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

np.set_string_function(lambda x: f'<np.array shape={x.shape} dtype={x.dtype}>')

Run = collections.namedtuple('Run', 'task method seed xs ys')

PALETTES = dict(
    defualt=(
        '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ),
    discrete=(
        '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
        '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    ),
    contrast=(
        '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
        '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    ),
    gradient=(
        '#fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d',
        '#46327e', '#440154',
    ),
    baselines=(
        '#222222', '#666666', '#aaaaaa', '#cccccc',
    ),
)

LEGEND = dict(
    fontsize=18,
    numpoints=1,
    labelspacing=0.25,
    columnspacing=1.2,
    handlelength=1.5,
    handletextpad=0.5,
    loc='upper center',
)

DEFAULT_BASELINES = ['d4pg', 'rainbow_sticky', 'human_gamer', 'impala', 'Optimal']


def find_keys(args):
  """Collect all scalar_data.jsonl files under data/reacherloca_fifo/*/phase_*/."""
  filenames = []
  for indir in args.indir:
    # indir is e.g. data/reacherloca_fifo
    for seed_dir in indir.iterdir():
      if not seed_dir.is_dir():
        continue
      for phase_name in ["phase_1", "phase_2", "phase_3"]:
        f = seed_dir / phase_name / "scalar_data.jsonl"
        if f.exists():
          filenames.append(f)

  keys = set()
  for filename in filenames:
    df = load_jsonl(filename)
    if df is None:
      continue
    keys |= set(df.columns)
  print(f"Keys      ({len(keys)}):", ", ".join(sorted(keys)), flush=True)


def load_runs(args):
  """One Run per seed. Each Run concatenates phase_1, phase_2, (phase_3) logs."""
  runs = []
  for indir in args.indir:
    # We'll use the directory name as the 'method' label, e.g. 'reacherloca_fifo'
    method = indir.name
    seed_dirs = sorted(d for d in indir.iterdir() if d.is_dir())
    total = len(seed_dirs)
    print(f"Found {total} seed dirs in {indir}")
    for seed_dir in seed_dirs:
      run = load_run(seed_dir, method, args)
      if run is not None:
        runs.append(run)
  return runs


def load_run(seed_dir, method, args):
  """
  seed_dir: data/reacherloca_fifo/0 (or 1,2,...)
  method:   e.g. 'reacherloca_fifo'
  We read phase_1/2/3 scalar_data.jsonl, concatenate, and bin.
  """
  task = "reacherloca"
  seed = seed_dir.name

  dfs = []
  for phase_name in ["phase_1", "phase_2", "phase_3"]:
    filename = seed_dir / phase_name / "scalar_data.jsonl"
    if not filename.exists():
      continue
    df = load_jsonl(filename)
    if df is None:
      continue
    dfs.append(df)

  if not dfs:
    print("Skipping empty run", task, method, seed)
    return

  df = pd.concat(dfs, ignore_index=True)

  try:
    df = df[[args.xaxis, args.yaxis]].dropna()
    if args.maxval:
      df = df.replace([+np.inf], +args.maxval)
      df = df.replace([-np.inf], -args.maxval)
      df[args.yaxis] = df[args.yaxis].clip(-args.maxval, +args.maxval)
  except KeyError:
    print("Missing xaxis/yaxis in", seed_dir)
    return

  xs = df[args.xaxis].to_numpy()
  if args.xmult != 1:
    xs = xs.astype(np.float32) * args.xmult
  ys = df[args.yaxis].to_numpy()

  # For Dreamer ReacherLoCA FIFO we can just pick a fixed bin size
  # (override with --bins if you like).
  if args.bins == -1:
    bins = 1e4  # default if not specified
  else:
    bins = args.bins

  if bins:
    borders = np.arange(0, xs.max() + 1e-8, bins)
    xs, ys = bin_scores(xs, ys, borders)

  if not len(xs):
    print("Skipping empty run after binning", task, method, seed)
    return

  return Run(task, method, seed, xs, ys)


def load_baselines(patterns, prefix=False):
  runs = []
  directory = pathlib.Path(__file__).parent.parent / 'scores'
  for filename in directory.glob('**/*_baselines.json'):
    for task, methods in json.loads(filename.read_text()).items():
      for method, score in methods.items():
        if prefix:
          method = f'baseline_{method}'
        if not any(p.search(method) for p in patterns):
          continue
        runs.append(Run(task, method, None, None, score))
  return runs


def stats(runs, baselines):
  tasks = sorted(set(r.task for r in runs))
  methods = sorted(set(r.method for r in runs))
  seeds = sorted(set(r.seed for r in runs))
  baseline = sorted(set(r.method for r in baselines))
  print('Loaded', len(runs), 'runs.')
  print(f'Tasks     ({len(tasks)}):', ', '.join(tasks))
  print(f'Methods   ({len(methods)}):', ', '.join(methods))
  print(f'Seeds     ({len(seeds)}):', ', '.join(seeds))
  print(f'Baselines ({len(baseline)}):', ', '.join(baseline))


def order_methods(runs, baselines, args):
  methods = []
  for pattern in args.methods:
    for method in sorted(set(r.method for r in runs)):
      if pattern.search(method):
        if method not in methods:
          methods.append(method)
        if method not in args.colors:
          index = len(args.colors) % len(args.palette)
          args.colors[method] = args.palette[index]
  non_baseline_colors = len(args.colors)
  for pattern in args.baselines:
    for method in sorted(set(r.method for r in baselines)):
      if pattern.search(method):
        if method not in methods:
          methods.append(method)
        if method not in args.colors:
          index = len(args.colors) - non_baseline_colors
          index = index % len(PALETTES['baselines'])
          args.colors[method] = PALETTES['baselines'][index]
  return methods


def figure(runs, methods, args):
  tasks = sorted(set(r.task for r in runs if r.xs is not None))
  rows = int(np.ceil((len(tasks) + len(args.add)) / args.cols))
  figsize = args.size[0] * args.cols, args.size[1] * rows
  fig, axes = plt.subplots(rows, args.cols, figsize=figsize, squeeze=False)
  for task, ax in zip(tasks, axes.flatten()):
    relevant = [r for r in runs if r.task == task]
    plot(task, ax, relevant, methods, args)
  for name, ax in zip(args.add, axes.flatten()[len(tasks):]):
    ax.set_facecolor((0.9, 0.9, 0.9))
    if name == 'median':
      plot_combined(
          'combined_median', ax, runs, methods, args,
          agg=lambda x: np.nanmedian(x, -1))
    elif name == 'mean':
      plot_combined(
          'combined_mean', ax, runs, methods, args,
          agg=lambda x: np.nanmean(x, -1))
    elif name == 'gamer_median':
      plot_combined(
          'combined_gamer_median', ax, runs, methods, args,
          lo='random', hi='human_gamer',
          agg=lambda x: np.nanmedian(x, -1))
    elif name == 'gamer_mean':
      plot_combined(
          'combined_gamer_mean', ax, runs, methods, args,
          lo='random', hi='human_gamer',
          agg=lambda x: np.nanmean(x, -1))
    elif name == 'record_mean':
      plot_combined(
          'combined_record_mean', ax, runs, methods, args,
          lo='random', hi='record',
          agg=lambda x: np.nanmean(x, -1))
    elif name == 'clip_record_mean':
      plot_combined(
          'combined_clipped_record_mean', ax, runs, methods, args,
          lo='random', hi='record', clip=True,
          agg=lambda x: np.nanmean(x, -1))
    elif name == 'seeds':
      plot_combined(
          'combined_seeds', ax, runs, methods, args,
          agg=lambda x: np.isfinite(x).sum(-1))
    elif name == 'human_above':
      plot_combined(
          'combined_above_human_gamer', ax, runs, methods, args,
          agg=lambda y: (y >= 1.0).astype(float).sum(-1))
    elif name == 'human_below':
      plot_combined(
          'combined_below_human_gamer', ax, runs, methods, args,
          agg=lambda y: (y <= 1.0).astype(float).sum(-1))
    else:
      raise NotImplementedError(name)
  if args.xlim:
    for ax in axes[:-1].flatten():
      ax.xaxis.get_offset_text().set_visible(False)
#   if args.xlabel:
#     for ax in axes[-1]:
#       ax.set_xlabel(args.xlabel)
#   if args.ylabel:
#     for ax in axes[:, 0]:
#       ax.set_ylabel(args.ylabel)
  for ax in axes.flatten()[len(tasks) + len(args.add):]:
    ax.axis('off')

  legend(fig, args.labels, ncol=2, **LEGEND, bbox_to_anchor=(0.5, 0.12))
  fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25)
  return fig


def plot(task, ax, runs, methods, args):
  assert runs
  try:
    title = task.split('_', 1)[1].replace('_', ' ').title()
  except IndexError:
    title = task.title()
  #ax.set_title(title)
  #ax.set_title("Phase 1", loc='left')
  #ax.set_title("Phase 2", loc='center')
  #ax.set_title("Phase 3", loc='right')
  xlim = [+np.inf, -np.inf]
  for index, method in enumerate(methods):
    relevant = [r for r in runs if r.method == method]
    if not relevant:
      continue
    if any(r.xs is None for r in relevant):
      baseline(index, method, ax, relevant, args)
    else:
      if args.agg == 'none':
        xs, ys = curve_lines(index, task, method, ax, relevant, args)
      else:
        xs, ys = curve_area(index, task, method, ax, relevant, args)
      if len(xs) == len(ys) == 0:
        print(f'Skipping empty: {task} {method}')
        continue
      xlim = [min(xlim[0], np.nanmin(xs)), max(xlim[1], np.nanmax(xs))]
  
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  steps = [1, 2, 2.5, 5, 10]
  ax.xaxis.set_major_locator(
      ticker.FixedLocator([5e5, 1e6, 1.5e6, 2e6, 2.5e6])
  )
  ax.yaxis.set_major_locator(ticker.MaxNLocator(args.yticks, steps=steps))

  if np.isfinite(xlim).all():
    ax.set_xlim(args.xlim or xlim)
  if args.xlim:
    ticks = sorted({*ax.get_xticks(), *args.xlim})
    ticks = [x for x in ticks if args.xlim[0] <= x <= args.xlim[1]]
    ax.set_xticks(ticks)
  if args.ylim:
    # If user passed explicit limits, respect them
    ax.set_ylim(args.ylim)
    if args.ylimticks:
      ticks = sorted({*ax.get_yticks(), *args.ylim})
      ticks = [x for x in ticks if args.ylim[0] <= x <= args.ylim[1]]
      ax.set_yticks(ticks)
  else:
    # Otherwise, force y-axis to start at 0 and keep auto upper limit
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)

  ax.text(0.5e6, 4000 + 80, "Phase 1", ha="center", va="bottom")
  ax.text(1.75e6, 2000 + 80, "Phase 2", ha="center", va="bottom")

  ax.grid(visible=True, axis='y', alpha=0.5, linestyle='--')
  ax.set_xlabel("Total Steps")
  ax.set_ylabel("Undiscounted Return")
#   ax.legend(fontsize=18, loc="best", ncol=1).set_zorder(100000)


def plot_combined(
    name, ax, runs, methods, args, agg, lo=None, hi=None, clip=False):
  tasks = sorted(set(run.task for run in runs if run.xs is not None))
  seeds = list(set(run.seed for run in runs))
  runs = [r for r in runs if r.task in tasks]  # Discard unused baselines.
  # Bin all runs onto the same X steps.
  borders = sorted(
      [r.xs for r in runs if r.xs is not None],
      key=lambda x: np.nanmax(x))[-1]
  for index, run in enumerate(runs):
    if run.xs is None:
      continue
    xs, ys = bin_scores(run.xs, run.ys, borders, fill='last')
    runs[index] = run._replace(xs=xs, ys=ys)
  # Per-task normalization by low and high baseline.
  if lo or hi:
    mins = collections.defaultdict(list)
    maxs = collections.defaultdict(list)
    [mins[r.task].append(r.ys) for r in load_baselines([re.compile(lo)])]
    [maxs[r.task].append(r.ys) for r in load_baselines([re.compile(hi)])]
    mins = {task: min(ys) for task, ys in mins.items() if task in tasks}
    maxs = {task: max(ys) for task, ys in maxs.items() if task in tasks}
    missing_baselines = []
    for task in tasks:
      if task not in mins or task not in maxs:
        missing_baselines.append(task)
    if set(missing_baselines) == set(tasks):
        print(f'No baselines found to normalize any tasks in {name} plot.')
    else:
      for task in missing_baselines:
        print(f'No baselines found to normalize {task} in {name} plot.')
    for index, run in enumerate(runs):
      if run.task not in mins or run.task not in maxs:
        continue
      ys = (run.ys - mins[run.task]) / (maxs[run.task] - mins[run.task])
      if clip:
        ys = np.minimum(ys, 1.0)
      runs[index] = run._replace(ys=ys)
  # Aggregate across tasks but not methods or seeds.
  combined = []
  for method, seed in itertools.product(methods, seeds):
    relevant = [r for r in runs if r.method == method and r.seed == seed]
    if not relevant:
      continue
    if relevant[0].xs is None:
      xs, ys = None, np.array([r.ys for r in relevant])
    else:
      xs, ys = stack_scores(*zip(*[(r.xs, r.ys) for r in relevant]))
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      combined.append(Run('combined', method, seed, xs, agg(ys)))
  plot(name, ax, combined, methods, args)


def curve_lines(index, task, method, ax, runs, args):
  zorder = 10000 - 10 * index - 1
  for run in runs:
    color = args.colors[method]
    ax.plot(run.xs, run.ys, label=method, color=color, zorder=zorder)
  xs, ys = stack_scores(*zip(*[(r.xs, r.ys) for r in runs]))
  return xs, ys


def curve_area(index, task, method, ax, runs, args):
  xs, ys = stack_scores(*zip(*[(r.xs, r.ys) for r in runs]))
  with warnings.catch_warnings():  # NaN buckets remain NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    if args.agg == 'std1':
      mean, std = np.nanmean(ys, -1), np.nanstd(ys, -1)
      # STE
      std = std / np.sqrt(ys.shape[1])
      lo, mi, hi = mean - std, mean, mean + std
    elif args.agg == 'per0':
      lo, mi, hi = [np.nanpercentile(ys, k, -1) for k in (0, 50, 100)]
    elif args.agg == 'per5':
      lo, mi, hi = [np.nanpercentile(ys, k, -1) for k in (5, 50, 95)]
    elif args.agg == 'per25':
      lo, mi, hi = [np.nanpercentile(ys, k, -1) for k in (25, 50, 75)]
    else:
      raise NotImplementedError(args.agg)
  color = args.colors[method]
  kw = dict(color=color, zorder=1000 - 10 * index, alpha=0.1, linewidths=0)
  mask = ~np.isnan(mi)
  xs, lo, mi, hi = xs[mask], lo[mask], mi[mask], hi[mask]
  ax.fill_between(xs, lo, hi, **kw)
  ax.plot(xs, mi, label=method, color=color, zorder=10000 - 10 * index - 1)
  return xs, mi


def baseline(index, method, ax, runs, args):
  assert all(run.xs is None for run in runs)
  ys = np.array([run.ys for run in runs])
  mean, std = ys.mean(), ys.std()
  color = args.colors[method]
  kw = dict(color=color, zorder=500 - 20 * index - 1, alpha=0.1, linewidths=0)
  ax.fill_between([-np.inf, np.inf], [mean - std] * 2, [mean + std] * 2, **kw)
  kw = dict( color=color, zorder=5000 - 10 * index - 1)
  ax.axhline(4000, label=method, xmin=0,   xmax=0.4, **kw)  # Phase 1 optimal
  ax.axhline(2000, label=None, xmin=0.4, xmax=1.0, **kw)  # Phase 2 optimal
  ax.axvline(1000000, label=None, linestyle='dotted', **kw)


def legend(fig, mapping=None, **kwargs):
    entries = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label.startswith('_'):
                continue
            if mapping and label in mapping:
                label = mapping[label]
            entries[label] = handle

    if not entries:
        return None

    fig.subplots_adjust(bottom=0.43, top=0.93)

    # Default placement: centered, just above the bottom margin
    bbox = kwargs.pop('bbox_to_anchor', (0.5, -0.04))
    loc = kwargs.pop('loc', 'upper center')

    leg = fig.legend(
        entries.values(),
        entries.keys(),
        bbox_to_anchor=bbox,
        loc=loc,
        **kwargs,
    )
    leg.get_frame().set_edgecolor('white')
    return leg


def save(fig, args):
  args.outdir.mkdir(parents=True, exist_ok=True)
  filename = args.outdir / 'curves.png'
  fig.savefig(filename, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
  print('Saved to', filename)
  filename = args.outdir / 'curves.pdf'
  fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
  try:
    subprocess.call(['pdfcrop', str(filename), str(filename)])
  except FileNotFoundError:
    print('Install texlive-extra-utils to crop PDF outputs.')


def bin_scores(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  with warnings.catch_warnings():  # Empty buckets become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    for start, stop in zip(borders[:-1], borders[1:]):
      left = (xs <= start).sum()
      right = (xs <= stop).sum()
      if left < right:
        value = reducer(ys[left:right])
      elif binned:
        value = {'nan': np.nan, 'last': binned[-1]}[fill]
      else:
        value = np.nan
      binned.append(value)
  return borders[1:], np.array(binned)


def stack_scores(multiple_xs, multiple_ys, fill='last'):
  longest_xs = sorted(multiple_xs, key=lambda x: len(x))[-1]
  multiple_padded_ys = []
  for xs, ys in zip(multiple_xs, multiple_ys):
    assert (longest_xs[:len(xs)] == xs).all(), (list(xs), list(longest_xs))
    value = {'nan': np.nan, 'last': ys[-1]}[fill]
    padding = [value] * (len(longest_xs) - len(xs))
    padded_ys = np.concatenate([ys, padding])
    multiple_padded_ys.append(padded_ys)
  stacked_ys = np.stack(multiple_padded_ys, -1)
  return longest_xs, stacked_ys


def load_jsonl(filename):
  try:
    with filename.open() as f:
      lines = list(f.readlines())
    records = []
    for index, line in enumerate(lines):
      try:
        records.append(json.loads(line))
      except Exception:
        if index == len(lines) - 1:
          continue  # Silently skip last line if it is incomplete.
        raise ValueError(
            f'Skipping invalid JSON line ({index+1}/{len(lines)+1}) in'
            f'{filename}: {line}')
    return pd.DataFrame(records)
  except ValueError as e:
    print('Invalid', filename, e)
    return None


def save_runs(runs, filename):
  filename.parent.mkdir(parents=True, exist_ok=True)
  records = []
  for run in runs:
    if run.xs is None:
      continue
    records.append(dict(
        task=run.task, method=run.method, seed=run.seed,
        xs=run.xs.tolist(), ys=run.ys.tolist()))
  runs = json.dumps(records)
  filename.write_text(runs)
  print('Saved', filename)


def main(args):
  find_keys(args)
  runs = load_runs(args)
  save_runs(runs, args.outdir / 'runs.json')
  
  tasks_in_data = sorted(set(r.task for r in runs))
  baselines = []
  for t in tasks_in_data:
      baselines.append(Run(t, "Optimal", None, None, np.array([0])))

  stats(runs, baselines)
  methods = order_methods(runs, baselines, args)
  if not runs:
    print('Noting to plot.')
    return
  # Adjust options based on loaded runs.
  tasks = set(r.task for r in runs)
  if 'auto' in args.add:
    index = args.add.index('auto')
    del args.add[index]
    atari = any(run.task.startswith('atari_') for run in runs)
    if len(tasks) < 2:
      pass
    elif atari:
      args.add[index:index] = [
          'gamer_median', 'gamer_mean', 'record_mean', 'clip_record_mean',
      ]
    else:
      args.add[index:index] = ['mean', 'median']
  args.cols = min(args.cols, len(tasks) + len(args.add))
  args.legendcols = min(args.legendcols, args.cols)
  print('Plotting...')
  fig = figure(runs + baselines, methods, args)
  save(fig, args)


def parse_args():
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--indir-prefix', type=pathlib.Path)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--subdir', type=boolean, default=True)
  parser.add_argument('--xaxis', type=str, default='step')
  parser.add_argument('--yaxis', type=str, default='eval_return')
  parser.add_argument('--tasks', nargs='+', default=[r'.*'])
  parser.add_argument('--methods', nargs='+', default=[r'.*'])
  parser.add_argument('--baselines', nargs='+', default=DEFAULT_BASELINES)
  parser.add_argument('--prefix', type=boolean, default=False)
  parser.add_argument('--bins', type=float, default=-1)
  parser.add_argument('--agg', type=str, default='std1')
  #parser.add_argument('--size', nargs=2, type=float, default=[6.25, 5.75])
  #parser.add_argument('--size', nargs=2, type=float, default=[9.25, 4.75])   
  parser.add_argument('--size', nargs=2, type=float, default=[7.25, 4.15])
  parser.add_argument('--dpi', type=int, default=80)
  parser.add_argument('--cols', type=int, default=6)
  parser.add_argument('--xlim', nargs=2, type=float, default=None)
  parser.add_argument('--ylim', nargs=2, type=float, default=None)
  parser.add_argument('--ylimticks', type=boolean, default=True)
  parser.add_argument('--xlabel', type=str, default=None)
  parser.add_argument('--ylabel', type=str, default=None)
  parser.add_argument('--xticks', type=int, default=6)
  parser.add_argument('--yticks', type=int, default=5)
  parser.add_argument('--xmult', type=float, default=1)
  parser.add_argument('--labels', nargs='+', default=None)
  parser.add_argument('--palette', nargs='+', default=['contrast'])
  parser.add_argument('--legendcols', type=int, default=4)
  parser.add_argument('--colors', nargs='+', default={})
  parser.add_argument('--maxval', type=float, default=0)
  parser.add_argument('--add', nargs='+', type=str, default=['auto', 'seeds'])
  args = parser.parse_args()
  if args.subdir:
    args.outdir /= args.indir[0].stem
  if args.indir_prefix:
    args.indir = [args.indir_prefix / indir for indir in args.indir]
  args.indir = [d.expanduser() for d in args.indir]
  args.outdir = args.outdir.expanduser()
  if args.labels:
    assert len(args.labels) % 2 == 0
    args.labels = {k: v for k, v in zip(args.labels[:-1], args.labels[1:])}
  if args.colors:
    assert len(args.colors) % 2 == 0
    args.colors = {k: v for k, v in zip(args.colors[:-1], args.colors[1:])}
  args.tasks = [re.compile(p) for p in args.tasks]
  args.methods = [re.compile(p) for p in args.methods]
  args.baselines = [re.compile(p) for p in args.baselines]
#   if 'return' not in args.yaxis:
#     args.baselines = []
  if args.prefix is None:
    args.prefix = len(args.indir) > 1
  if len(args.palette) == 1 and args.palette[0] in PALETTES:
    args.palette = 10 * PALETTES[args.palette[0]]
  if len(args.add) == 1 and args.add[0] == 'none':
    args.add = []
  return args


if __name__ == '__main__':
  main(parse_args())