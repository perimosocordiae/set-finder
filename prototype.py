import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from itertools import combinations
from argparse import ArgumentParser

# color tuples
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def _main_static(imgfile, debug=False, text=False, **kwargs):
  img = cv2.imread(imgfile)
  if img is None:
    print 'Error: could not read %s' % imgfile
    return
  rects, attrs = process_image(img, **kwargs)
  if text:
    print imgfile, [' '.join(map(str, attr)) for attr in attrs]
  elif debug:
    win_name = 'debug view for ' + imgfile
    show_debug_view(img, rects, attrs, win_name=win_name)
  else:
    win_name = 'set view for ' + imgfile
    show_set_view(img, rects, attrs, win_name=win_name)


def _main_camera(debug=False, text=False, updown_key='', lr_key='',
                 page_key='', **kwargs):
  show_fn = show_debug_view if debug else show_set_view
  win_name = 'debug viewer' if debug else 'set viewer'
  cv2.namedWindow(win_name)
  vc = cv2.VideoCapture(0)

  if not vc.isOpened():
    print "Couldn't open a webcam for frame capture"
    return

  key_control = {
      2621440: (updown_key, -1),
      2490368: (updown_key, +1),
      2424832: (lr_key, -1),
      2555904: (lr_key, +1),
      2228224: (page_key, -1),
      2162688: (page_key, +1),
  }
  # remove any inactive keys
  for key, (param, _) in key_control.items():
    if param not in kwargs:
      del key_control[key]

  print 'Press escape to quit'
  got_frame, img = vc.read()
  while got_frame:
    rects, attrs = process_image(img, **kwargs)
    key = show_fn(img, rects, attrs, frame_delay=1, win_name=win_name)
    if key == 27:  # ESC
      break
    elif key in key_control:
      param, delta = key_control[key]
      if delta < 0:
        kwargs[param] = max(kwargs[param] + delta, 0)
      else:
        kwargs[param] = min(kwargs[param] + delta, 255)
      print param, kwargs[param]
    got_frame, img = vc.read(img)

  cv2.destroyWindow(win_name)


def process_image(img, max_dim=800, **kwargs):
  # rescale if necessary
  scale = float(max_dim)/max(img.shape[:2])
  if scale < 1:
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA)

  rects, attrs = process_cards(img, **kwargs)
  # rescale rects to match original image scale
  if scale < 1:
    for rect in rects:
      rect /= scale
  # order rects by x-position
  centers = np.array([cv2.minEnclosingCircle(c)[0] for c in rects])
  order = np.argsort(centers.dot([1000, 1]))  # hacky sort by x-position
  rects = [rects[i] for i in order]
  attrs = [attrs[i] for i in order]
  return rects, attrs


def process_cards(img, side_err_scale=0.02, min_area=1000,
                  max_corner_angle_cos=0.3, **kwargs):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # start by finding card rectangles in hsv space
  metric = hsv[:,:,1].astype(float) * hsv[:,:,2].astype(float)
  metric /= metric.max()
  metric *= 255
  metric = metric.astype(np.uint8)
  # Otsu thresholding to split the white part of cards from non-cards
  _, mask = cv2.threshold(metric, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)

  rects = []
  attrs = []
  for cnt in contours:
    side_err_thresh = side_err_scale * cv2.arcLength(cnt, True)
    # approximate the contour with fewer edges
    cnt = cv2.approxPolyDP(cnt, side_err_thresh, True)[:,0]  # <- squeeze
    if not (  # TODO: check convexityDefects instead of isContourConvex
            len(cnt) == 4 and
            cv2.contourArea(cnt) > min_area and
            cv2.isContourConvex(cnt) and
            angle_cos(cnt).max() < max_corner_angle_cos):
      continue
    # now find the attributes for this card
    attr = process_one_card(img, cnt, **kwargs)
    if attr is not None:
      rects.append(cnt)
      attrs.append(attr)
  return rects, attrs


def process_one_card(img, rect, card_width=450, card_height=450, **kwargs):
  # Crop the card out of the overall image
  h = np.array([[0,0],[card_width-1,0],
                [card_width-1,card_height-1],[0,card_height-1]], np.float32)
  transform = cv2.getPerspectiveTransform(rect.astype(np.float32), h)
  card = cv2.warpPerspective(img, transform, (card_width, card_height))
  # Return the card's attributes
  return process_attributes(card, **kwargs)


def process_attributes(card, min_shape_area=0.05, max_shape_area=0.9,
                       shape_close=10, **kwargs):
  hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

  # find the shapes, thresholding on high-saturation, low-value pixels
  metric = hsv[:,:,1] - hsv[:,:,2]
  # histogram(metric) is bimodal, so Otsu thresholding is ideal
  _, mask = cv2.threshold(metric, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # optionally do a morphological close on the mask, to fill in some gaps
  if shape_close > 0:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shape_close,shape_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
  contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

  if not contours:
    return

  # filter out contours outside the bounds (as a fraction of total card area)
  dims = np.array([cv2.boundingRect(c)[2:] for c in contours], dtype=float)
  dims /= card.shape[:2]  # scale to [0,1] in card dimensions
  areas = np.product(dims, axis=1)
  contours = [c for a,c in zip(areas, contours)
              if min_shape_area < a < max_shape_area]

  if not contours:
    return

  # hack: use drawContours to make a mask of in-contour pixels
  mask[...] = 0
  cv2.drawContours(mask, contours, -1, 255, -1)

  # find the color (red, green, purple)
  color = card_color(card, hsv, mask)

  # XXX: re-finding the contours here
  _, contours, hier = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
  outer_mask = hier[0,:,-1] < 0
  filling = card_filling(outer_mask)
  outer_contours = [c for i,c in enumerate(contours) if outer_mask[i]]
  shape = card_shape(outer_contours)
  return len(outer_contours), filling, color, shape


def show_debug_view(img, rects, attrs, frame_delay=-1, win_name=''):
  cv2.drawContours(img, rects, -1, GREEN, 3)
  for idx, (rect, attr) in enumerate(zip(rects, attrs)):
    label = ''.join(str(a)[:3].title() for a in attr)
    add_text(img, label, rect.min(axis=0), scale=0.5)
    add_text(img, str(idx+1), rect.mean(axis=0).astype(int) + (-5,5),
             fgcolor=BLACK, bgcolor=None, thickness=3)
  cv2.imshow(win_name, img)
  return cv2.waitKey(frame_delay)


def show_set_view(img, rects, attrs, frame_delay=-1, win_name=''):
  for i,j,k in find_sets(attrs):
    cv2.drawContours(img, (rects[i],rects[j],rects[k]), -1, GREEN, 3)
    break
  else:
    pos = (img.shape[0]/2, img.shape[1]/2 - 100)
    add_text(img, "No sets found", pos, fgcolor=RED, thickness=2)
  cv2.imshow(win_name, img)
  return cv2.waitKey(frame_delay)


def add_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1,
             bgcolor=BLACK, fgcolor=WHITE, thickness=1):
  fgpos = tuple(pos)
  if bgcolor is not None:
    bgpos = (fgpos[0] + thickness, fgpos[1] + thickness)
    cv2.putText(img, text, bgpos, font, scale, bgcolor, thickness)
  cv2.putText(img, text, fgpos, font, scale, fgcolor, thickness)


def find_sets(attributes):
  for i, j, k in combinations(xrange(len(attributes)), 3):
    for items in zip(attributes[i],attributes[j],attributes[k]):
      if len(set(items)) not in (1,3):
        break
    else:
      yield i,j,k


def card_filling(outer_mask):
  num_contours = len(outer_mask)
  num_outers = outer_mask.sum()
  if num_contours == num_outers:
    # all contours are outers
    return 'solid'
  if num_contours//2 == num_outers:
    # there's one inner for every outer
    return 'open'
  return 'striped'


def card_shape(contours, side_err_scale=0.01):
  for contour in contours:
    al = cv2.arcLength(contour, True)
    al_rect = sum(cv2.boundingRect(contour)[2:]) * 2
    if al / al_rect > 1:
      continue  # very concave!
    approx = cv2.approxPolyDP(contour, side_err_scale * al, True)
    if 4 <= len(approx) <= 8:  # a little wiggle room
      return 'diamond'
    if cv2.isContourConvex(approx):
      return 'oval'
    return 'ess'
  # all contours are very convex: try something else!
  for contour in contours:
    hull = cv2.convexHull(contour)
    al = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, side_err_scale*al, True)
    if 4 <= len(approx) <= 6:
      return 'diamond'
  return 'oval'  # or ess, not really sure here


def card_color(card, hsv, thresh):
  in_contour = hsv[thresh!=0][None]
  # magic constants! (min_hue, max_hue, min_sat)
  # TODO: replace this with a distribution-matching method
  ranges = [(25, 110, 0), (120, 255, 0), (0, 10, 60)]
  names = ['green', 'purple', 'red']
  match_score = np.zeros(3)
  for i, (min_hue, max_hue, min_sat) in enumerate(ranges):
    mask = cv2.inRange(in_contour, (min_hue, min_sat, 0), (max_hue, 255, 255))
    # score by value when in the correct hue/sat range
    # idea: low-value pixels are washed out, thus unreliable
    match_score[i] = in_contour[mask>0,2].sum()
  ci = np.argmax(match_score)
  return names[ci]


def angle_cos(contour):
  # hack in a wrap-around diff
  d = np.diff(np.pad(contour, ((0,2),(0,0)), mode='wrap'), axis=0)
  # compute normalized angles
  norm = np.linalg.norm(d, axis=1)
  return np.abs(inner1d(d[:-1], d[1:])) / (norm[:-1]*norm[1:])


def parse_args():
  ap = ArgumentParser()

  ag = ap.add_argument_group('Input Arguments')
  ag.add_argument('files', metavar='file', nargs='*',
                  help='Input image file(s).')
  ag.add_argument('--camera', action='store_true',
                  help='Use webcam input instead of static files.')

  ap.add_argument('--debug', action='store_true')
  ap.add_argument('--text', action='store_true', help='Display text output')

  ag = ap.add_argument_group('Card Detection Parameters')
  ag.add_argument('--side-error-scale', type=float, default=0.02)
  ag.add_argument('--min-area', type=int, default=1000)
  ag.add_argument('--max-corner-angle-cos', type=float, default=0.3)

  ag.add_argument_group('Attribute Detection Parameters')
  ag.add_argument('--min-shape-area', type=float, default=0.05)
  ag.add_argument('--max-shape-area', type=float, default=0.9)
  ag.add_argument('--shape-close', type=int, default=0)

  ag = ap.add_argument_group('Internal Parameters')
  ag.add_argument('--max-dim', type=int, default=800)
  ag.add_argument('--card-width', type=int, default=450)
  ag.add_argument('--card-height', type=int, default=450)

  ag = ap.add_argument_group('Camera-mode Key Bindings')
  ag.add_argument('--updown-key', default='max_sat')
  ag.add_argument('--lr-key', default='min_val')
  ag.add_argument('--page-key', default='min_gray')

  args = ap.parse_args()
  if not (args.files or args.camera):
    ap.error('too few arguments')
  if args.files and args.camera:
    ap.error('cannot use both static files and webcam input')
  return args


def main(camera=False, files=None, **kwargs):
  if camera:
    _main_camera(**kwargs)
  else:
    for f in files:
      _main_static(f, **kwargs)


# for debug purposes only
def hist_lines(im, arr, color=RED):
  h = im.shape[0]
  hist_item = cv2.calcHist([arr],[0],None,[256],[0,256])
  cv2.normalize(hist_item,hist_item,0,h,cv2.NORM_MINMAX)
  hist=np.int32(np.around(hist_item))
  for x,y in enumerate(hist):
    cv2.line(im,(x,h),(x,h-y), color)


if __name__ == '__main__':
  main(**vars(parse_args()))
