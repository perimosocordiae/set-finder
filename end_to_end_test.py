import sys
import ast
from cStringIO import StringIO
from prototype import _main_static

expected_out = {
    'test_images/set01.png': [
        '3 open purple oval', '1 solid red diamond', '2 solid green oval',
        '2 open green diamond', '2 solid red diamond', '1 striped red ess',
        '1 open purple diamond', '3 striped red oval', '2 striped green oval',
        '1 striped red diamond', '3 open green ess', '3 striped green oval'],
    'test_images/set02.png': [
        '1 solid red diamond', '1 striped red ess', '3 open green ess',
        '1 open purple diamond', '2 solid red diamond', '1 striped red diamond',
        '3 open purple oval', '2 striped green oval', '2 open green diamond',
        '3 striped red oval', '2 solid green oval', '3 striped green oval'],
    'test_images/set03.png': [
        '3 striped green oval', '1 striped red ess', '3 open green ess',
        '2 striped green oval', '3 striped red oval', '1 open purple diamond',
        '1 striped red ess', '2 solid red diamond', '2 open green diamond',
        '1 solid red diamond', '2 solid green oval', '3 open purple oval'],
    'test_images/set04.png': [
        '3 striped green oval', '3 striped red oval', '1 striped red diamond',
        '2 open green diamond', '2 striped green oval', '2 solid green oval',
        '3 open green ess', '2 solid red diamond', '1 open purple diamond',
        '3 open purple oval', '1 striped red ess', '1 solid red diamond'],
    'test_images/set05.png': [
        '2 solid green oval', '3 striped green oval', '3 striped red oval',
        '2 open green diamond', '2 striped green oval', '3 open purple oval',
        '2 solid red diamond', '1 striped red diamond', '3 open green ess',
        '1 open purple diamond', '1 striped red ess', '1 solid red diamond'],
}


def main():
  real_stdout = sys.stdout
  for filename, expected in expected_out.iteritems():
    test_out = StringIO()
    sys.stdout = test_out
    _main_static(filename, text=True)
    sys.stdout = real_stdout
    _, result = test_out.getvalue().strip().split(' ', 1)
    actual = ast.literal_eval(result)
    if len(actual) != len(expected):
      print 'Test failure on %s:' % filename
      print ' mismatching number of cards:',
      print 'expected %d, got %d' % (len(expected), len(actual))
      break
    mismatches = [(a,e) for a,e in zip(actual, expected) if a != e]
    if mismatches:
      print 'Test failure on %s:' % filename
      for mm in mismatches:
        print ' %s != %s' % mm
      break
  else:
    print 'All tests passed!'

if __name__ == '__main__':
  main()
