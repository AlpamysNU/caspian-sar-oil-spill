"""Manual label review in napari (2026-10-04). Edits a REVISED copy only; the original dataset is never written.

Run in an environment with napari:  python -m segmentation.label_review.review_napari scenes.txt
scenes.txt: one scene per line (img_0936 or train_img_0936). Output (default <project root>/results/label_review/revised_manual): labels_1D_revised/<name>.png (only scenes
that differ from the original), overlays/<name>.png (before/after audit image), revision_log_manual.csv (one row per save).

Screen: SAR image | 'original (outline)' = read-only reference, drawn as contours | 'revised' = the editable class map (0 sea, 1 oil cyan, 2 look-alike red,
3 ship orange, 4 land green). Paint in 'revised' with the napari Labels tools (p paint, e erase, f fill, polygon tool in the left toolbar); keys 0-4 choose the class
to paint with. Right dock: choose a category and an optional note, then Save + next / Previous / Reset scene. Everything is in array coordinates (1 screen pixel
at maximum zoom = 1 mask pixel), no resizing; each save is validated (same shape, uint8, values 0..4) before it is written.
"""
import argparse
from pathlib import Path

import napari
import numpy as np
from PIL import Image
from magicgui import magicgui
from magicgui.widgets import Container, ComboBox, Label, LineEdit, PushButton

from segmentation import paths
from segmentation.label_review import review_logic as rl


def color_dict(alpha):
    d = {0: (0, 0, 0, 0)}
    d.update({k: (c[0] / 255, c[1] / 255, c[2] / 255, alpha) for k, c in rl.COLORS.items()})
    return d


def add_labels(viewer, data, name, alpha, **kw):
    cd = color_dict(alpha)
    try:   # napari >= 0.5
        from napari.utils.colormaps import DirectLabelColormap
        return viewer.add_labels(data, name=name, colormap=DirectLabelColormap(color_dict={**cd, None: (0, 0, 0, 0)}), **kw)
    except Exception:
        return viewer.add_labels(data, name=name, color=cd, **kw)


class Reviewer:
    """napari session over a list of scenes; all file logic lives in review_logic."""

    def __init__(self, scenes, root, out, proposals=None, start=None):
        self.scenes, self.root, self.out, self.proposals = scenes, root, out, proposals
        out.mkdir(parents=True, exist_ok=True)
        self.log = out / 'revision_log_manual.csv'
        self.i, self.name, self.image, self.original = 0, None, None, None
        self.viewer = napari.Viewer(title='label review')
        self.status = Label(value='')
        self.category = ComboBox(label='category', choices=rl.CATEGORIES, value='no_change')
        self.note = LineEdit(label='note')
        b_save, b_prev, b_reset = PushButton(text='Save + next'), PushButton(text='Previous (save)'), PushButton(text='Reset scene')
        b_check, b_use = PushButton(text='Check changes (no save)'), PushButton(text='Use proposal (copy into revised)')
        self.goto_box, b_go = LineEdit(label='go to scene'), PushButton(text='Go (saves this scene first)')
        self.fix_to = ComboBox(label='replace invalid values with class', choices=[('0 sea', 0), ('1 oil', 1), ('2 look-alike', 2), ('3 ship', 3), ('4 land', 4)], value=0)
        b_fix = PushButton(text='Fix invalid values (>4)')
        self.viewer.window.add_dock_widget(Container(widgets=[self.status, self.category, self.note, b_check, b_use, b_save, b_prev, b_reset, self.goto_box, b_go, self.fix_to, b_fix]),
                                           area='right', name='review')
        b_save.clicked.connect(lambda: self.go(1))
        b_prev.clicked.connect(lambda: self.go(-1))
        b_reset.clicked.connect(lambda: self.load(self.i))
        b_check.clicked.connect(self.check)
        b_go.clicked.connect(self.goto)
        b_fix.clicked.connect(self.fix_invalid)
        b_use.clicked.connect(self.use_proposal)
        for key in '01234':
            def _select(layer, k=int(key)):
                self.viewer.layers['revised'].selected_label = k
            self.viewer.bind_key(key, _select, overwrite=True)
        self.load(scenes.index(start) if start in scenes else 0)

    def load(self, i):
        name = self.scenes[i]
        image, original, revised = rl.load_scene(self.root, name, self.out)
        self.i, self.name, self.image, self.original = i, name, image, original
        v = self.viewer
        v.layers.clear()
        v.add_image(image, name='SAR', colormap='gray', contrast_limits=(0, 255))
        ref = add_labels(v, original.copy(), 'original (outline)', 1.0)
        ref.contour, ref.editable = 2, False
        prop = self.proposals / f'{name}.png' if self.proposals else None
        if prop is not None and prop.exists():
            pl = add_labels(v, np.asarray(Image.open(prop)).astype(np.uint8), 'proposal (outline, read-only)', 1.0)
            pl.contour, pl.editable = 2, False
        rev = add_labels(v, revised.copy(), 'revised', 0.45)
        v.layers.selection.active = rev
        rev.mode, rev.selected_label = 'paint', 3
        self._last_valid = 3
        rev.events.selected_label.connect(self._clamp_label)
        self.note.value, self.category.value = '', 'no_change'
        self.status.value = f'{name}   [{i + 1}/{len(self.scenes)}]   {image.shape[1]}x{image.shape[0]} px'

    def _clamp_label(self, event=None):
        """napari can raise the brush number above 4 (spin-box arrow, 'M' = new label); keep it inside the five classes."""
        layer = self.viewer.layers['revised'] if 'revised' in self.viewer.layers else None
        if layer is None:
            return
        k = int(layer.selected_label)
        if 0 <= k <= 4:
            self._last_valid = k
        else:
            layer.selected_label = self._last_valid
            self.status.value = f'brush number {k} is not a class (valid 0-4); reset to {self._last_valid}'

    def fix_invalid(self):
        layer = self.viewer.layers['revised']
        d = np.asarray(layer.data).copy()
        bad = d > 4
        n = int(bad.sum())
        if n == 0:
            self.status.value = 'no invalid values in "revised"'
            return
        d[bad] = int(self.fix_to.value)
        layer.data = d.astype(np.uint8)
        self.status.value = f'{n} px with values >4 were set to class {int(self.fix_to.value)}; check the result and press Save + next'

    def extra_painted_layers(self):
        """Labels layers other than the two managed ones that contain painted pixels (a common cause of '0 px changed')."""
        keep = ('original (outline)', 'revised', 'proposal (outline, read-only)')
        return [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Labels) and l.name not in keep and np.asarray(l.data).any()]

    def check(self):
        rev = np.asarray(self.viewer.layers['revised'].data).astype(np.uint8)
        tr = rl.transitions(self.original, rev)
        extra = self.extra_painted_layers()
        self.status.value = (f'{self.name}: {int((rev != self.original).sum())} px differ from the original ' + (str(tr) if tr else '(nothing painted in "revised")')
                             + (f'   PAINTED IN OTHER LAYER(S): {extra}' if extra else ''))

    def use_proposal(self):
        prop = self.proposals / f'{self.name}.png' if self.proposals else None
        if prop is None or not prop.exists():
            self.status.value = 'no proposal for this scene'
            return
        self.viewer.layers['revised'].data = np.asarray(Image.open(prop)).astype(np.uint8)
        self.status.value = f'{self.name}: proposal copied into "revised"; edit it and press Save + next'

    def save_current(self):
        rev = np.asarray(self.viewer.layers['revised'].data).astype(np.uint8)
        extra = self.extra_painted_layers()
        if extra:
            self.status.value = f'NOT SAVED: pixels were painted in layer(s) {extra}, not in "revised". Paint in "revised" (select it in the layer list).'
            return False
        if int((rev != self.original).sum()) > 0 and self.category.value == 'no_change':
            self.status.value = 'NOT SAVED: pixels differ from the original but the category is "no_change"; choose the category that describes the change (ship_added, class_fixed, ...).'
            return False
        if int((rev != self.original).sum()) == 0 and self.category.value not in ('no_change', 'ambiguous'):
            self.status.value = f'NOT SAVED: category is "{self.category.value}" but "revised" equals the original (nothing painted in "revised"). Press "Check changes".'
            return False
        try:
            changed, warnings = rl.save(self.root, self.out, self.name, self.image, self.original, rev, self.category.value,
                                        self.note.value, self.log)
        except ValueError as e:
            self.status.value = f'NOT SAVED: {e}'
            return False
        self.status.value = f'saved {self.name}: {changed} px changed' + (f'  WARNING: {"; ".join(warnings)}' if warnings else '')
        return True

    def goto(self):
        name = rl.scene_name(self.goto_box.value)
        if name not in self.scenes:
            self.status.value = f'{name} is not in this list ({len(self.scenes)} scenes); first {self.scenes[0]}, last {self.scenes[-1]}'
            return
        if self.save_current():
            self.load(self.scenes.index(name))

    def go(self, step):
        if self.save_current():
            j = self.i + step
            if 0 <= j < len(self.scenes):
                self.load(j)
            else:
                self.status.value += '  (end of the list)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_root_argument(ap)
    ap.add_argument('scenes', type=Path)
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/revised_manual')
    ap.add_argument('--proposals', type=Path, default=None, help='folder with <scene>.png proposals (1D maps); shown as a read-only outline layer, copied by the button')
    ap.add_argument('--start', default=None, help='scene to open first (img_XXXX), default the first of the list')
    a = ap.parse_args()
    Reviewer([rl.scene_name(s) for s in a.scenes.read_text().split() if s.strip()], a.root, a.out, a.proposals,
             rl.scene_name(a.start) if a.start else None)
    napari.run()


if __name__ == '__main__':
    main()
