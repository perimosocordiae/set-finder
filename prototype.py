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


def _main_static(imgfile, debug=False):
  img = cv2.imread(imgfile)
  if img is None:
    print 'Error: could not read %s' % imgfile
    return
  img, rects, attrs = process_image(img)
  if debug:
    show_debug_view(img, rects, attrs, name=imgfile)
  else:
    show_set_view(img, rects, attrs, name=imgfile)


def _main_camera(debug=False):
  show_fn = show_debug_view if debug else show_set_view
  win_name = 'debug viewer' if debug else 'set viewer'
  cv2.namedWindow(win_name)
  vc = cv2.VideoCapture(0)

  if not vc.isOpened():
    print "Couldn't open a webcam for frame capture"
    return

  print 'Press escape to quit'
  got_frame, img = vc.read()
  while got_frame:
    img, rects, attrs = process_image(img)
    key = show_fn(img, rects, attrs, frame_delay=1, win_name=win_name)
    if key == 27:  # ESC
      break
    got_frame, img = vc.read(img)

  cv2.destroyWindow(win_name)


def process_image(img):
  # rescale if necessary
  scale = 800./max(img.shape[:2])
  if scale < 1:
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA)

  rects = find_rects(img)
  attrs = filter(None, [attributes(crop_card(img, r)) for r in rects])
  return img, rects, attrs


def show_debug_view(img, rects, attrs, frame_delay=-1, win_name=''):
  cv2.drawContours(img, rects, -1, GREEN, 3)
  for rect, attr in zip(rects, attrs):
    label = ''.join(str(a)[:3].title() for a in attr)
    add_text(img, label, rect.min(axis=0), scale=0.5)
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
  bgpos = (fgpos[0] + thickness, fgpos[1] + thickness)
  cv2.putText(img, text, bgpos, font, scale, bgcolor, thickness)
  cv2.putText(img, text, fgpos, font, scale, fgcolor, thickness)


def crop_card(img, bbox, width=450, height=450):
  bbox = bbox.astype(np.float32)
  h = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], np.float32)
  transform = cv2.getPerspectiveTransform(bbox, h)
  return cv2.warpPerspective(img, transform, (width, height))


def find_sets(attributes):
  for i, j, k in combinations(xrange(len(attributes)), 3):
    for items in zip(attributes[i],attributes[j],attributes[k]):
      if len(set(items)) not in (1,3):
        break
    else:
      yield i,j,k


def attributes(card):
  hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

  # find the shapes, thresholding on high-value pixels
  val = hsv[:,:,2]
  mean_val = val.mean()
  thresh = (val > mean_val).astype(np.uint8) * 255
  contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

  # filter out contours less than 5% or more than 90% of total card area
  dims = np.array([cv2.boundingRect(c)[2:] for c in contours], dtype=float)
  dims /= thresh.shape  # scale to [0,1] in card dimensions
  areas = np.product(dims, axis=1)
  contours = [c for a,c in zip(areas, contours) if 0.05 < a < 0.9]

  if not contours:
    return

  # hack: use drawContours to make a mask of in-contour pixels
  thresh[...] = 0
  cv2.drawContours(thresh, contours, -1, 255, -1)
  thresh = cv2.erode(thresh,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))
  # find the color (red, green, purple)
  color = card_color(card, hsv, thresh)

  # XXX: re-finding the contours here
  _, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
  outer_mask = hier[0,:,-1] < 0
  filling = card_filling(outer_mask)
  outer_contours = [c for i,c in enumerate(contours) if outer_mask[i]]
  shape = card_shape(outer_contours)
  return len(outer_contours), filling, color, shape


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


def find_rects(img, min_val=190, max_sat=130, min_gray=90,
               side_err_scale=0.02, min_area=1000, max_corner_angle_cos=0.3):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  thresh = ((hsv[:,:,1] <= max_sat) &
            (gray >= min_gray) &
            (hsv[:,:,2] >= min_val)).astype(np.uint8)*255

  _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
  rects = []
  for cnt in contours:
    side_err_thresh = side_err_scale * cv2.arcLength(cnt, True)
    # approximate the contour with fewer edges
    cnt = cv2.approxPolyDP(cnt, side_err_thresh, True)[:,0]  # <- squeeze
    if (  # TODO: check convexityDefects instead of isContourConvex
            len(cnt) == 4 and
            cv2.contourArea(cnt) > min_area and
            cv2.isContourConvex(cnt) and
            angle_cos(cnt).max() < max_corner_angle_cos):
      rects.append(cnt)
  return rects


def main():
  ap = ArgumentParser()
  ap.add_argument('--camera', action='store_true', help='Use webcam input')
  ap.add_argument('--debug', action='store_true')
  ap.add_argument('file', nargs='*', help='Input image file(s).')
  args = ap.parse_args()
  if args.camera:
    _main_camera(debug=args.debug)
  elif not args.file:
    ap.error('too few arguments')
  else:
    for f in args.file:
      _main_static(f, debug=args.debug)


if __name__ == '__main__':
  main()
