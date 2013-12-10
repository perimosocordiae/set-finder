
import cv2
import numpy as np
from itertools import combinations


def main(debug=True):
  imgfile = 'test_images/set02.jpg'
  img = cv2.imread(imgfile)
  # rescale if neccessary
  scale = 800./max(img.shape[:2])
  if scale < 1:
    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

  rects = find_rects(img)
  attrs = [attributes(crop_card(img, r)) for r in rects]

  if debug:
    print "Found %d cards" % len(rects)
    cv2.drawContours(img, rects, -1, (0, 255, 0), 3)
    for rect, attr in zip(rects, attrs):
      label = ''.join(str(a)[:3].title() for a in attr)
      pos = rect.min(axis=0)
      cv2.putText(img, label, tuple(pos+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
      cv2.putText(img, label, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imshow('rects', img)
    cv2.waitKey()
  else:
    for i,j,k in find_sets(attrs):
      cv2.drawContours(img, (rects[i],rects[j],rects[k]), -1, (0,255,0), 3)
      break
    else:
      pos = (img.shape[0]/2, img.shape[1]/2 - 100)
      cv2.putText(img, "No sets found", (pos[0]+2,pos[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
      cv2.putText(img, "No sets found", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('set', img)
    cv2.waitKey()


def crop_card(img, bbox, width=450, height=450):
  bbox = bbox.astype(np.float32)
  h = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], np.float32)
  transform = cv2.getPerspectiveTransform(bbox, h)
  return cv2.warpPerspective(img, transform, (width,height))


def find_sets(attributes):
  for i,j,k in combinations(xrange(len(attributes)), 3):
    for items in zip(attributes[i],attributes[j],attributes[k]):
      if len(set(items)) not in (1,3):
        break
    else:
      yield i,j,k


def attributes(card, min_saturation=50, min_value=90, min_area=100, use_kmeans=False):
  hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
  color = card_color(hsv, min_saturation, min_value)

  # threshold out the shapes
  if use_kmeans:
    sat = hsv[:,:,1].ravel()
    sat = sat.astype(np.float32)[:,None]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,labels,centers = cv2.kmeans(sat, 2, criteria, 10, cv2.KMEANS_PP_CENTERS)
    thresh = (labels == np.argmax(centers)).reshape((hsv.shape[0], hsv.shape[1])).astype(np.uint8) * 255
  else:
    thresh = cv2.inRange(hsv, (0,50,0), (255,255,255))

  filling = card_filling(hsv, thresh)
  contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  mask = hier[0,:,-1] < 0
  contours = [c for i,c in enumerate(contours) if mask[i] and cv2.contourArea(c) > min_area]
  shape = card_shape(contours)
  return len(contours), filling, color, shape


def card_filling(hsv, thresh):
  _,saturation,value = hsv[thresh.astype(bool)].T
  ms = np.median(saturation)
  mv = np.median(value)
  if ms > mv:
    return 'solid'
  if mv - ms < 60:
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
    return 'squiggle'
  # all contours are very convex: try something else!
  for contour in contours:
    hull = cv2.convexHull(contour)
    al = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, side_err_scale*al, True)
    if 4 <= len(approx) <= 6:
      return 'diamond'
  return 'oval'  # or squiggle, not really sure here


def card_color(hsv, min_saturation, min_value):
  hue_ranges = [(60,110),(120, 140),(160, 200)]
  hue_names = ['green', 'purple', 'red']
  hue_matches = np.zeros(3)
  for i, hr in enumerate(hue_ranges):
    lb = (hr[0],min_saturation,min_value)
    hue_matches[i] = cv2.inRange(hsv, lb, (hr[1],255,255)).sum()
  return hue_names[np.argmax(hue_matches)]


def angle_cos(contour):
  p1 = np.roll(contour, -1, axis=0)
  d1 = contour - p1
  d2 = np.roll(contour, -2, axis=0) - p1
  return np.abs((d1*d2).sum(axis=1) / np.sqrt((d1*d1).sum(axis=1)*(d2*d2).sum(axis=1)))


def find_rects(img, side_err_scale=0.02, min_area=1000,  max_corner_angle_cos=0.3):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  thresh = cv2.inRange(hsv, (0,0,150),(255,50,255))
  contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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


if __name__ == '__main__':
  main()
