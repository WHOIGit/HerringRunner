import datetime
import re


# From django.utils.dateparse
standard_duration_re = re.compile(
  r'^'
  r'(?:(?P<days>-?\d+) (days?, )?)?'
  r'(?P<sign>-?)'
  r'((?:(?P<hours>\d+):)(?=\d+:\d+))?'
  r'(?:(?P<minutes>\d+):)?'
  r'(?P<seconds>\d+)'
  r'(?:\.(?P<microseconds>\d{1,6})\d{0,6})?'
  r'$'
)
def parse_duration(value):
  match = standard_duration_re.match(value)
  if match:
    kw = match.groupdict()
    days = datetime.timedelta(float(kw.pop('days', 0) or 0))
    sign = -1 if kw.pop('sign', '+') == '-' else 1
    if kw.get('microseconds'):
      kw['microseconds'] = kw['microseconds'].ljust(6, '0')
    if kw.get('seconds') and kw.get('microseconds'):
      if kw['seconds'].startswith('-'):
        kw['microseconds'] = '-' + kw['microseconds']
    kw = {k: float(v) for k, v in kw.items() if v is not None}
    return days + sign * datetime.timedelta(**kw)

def jsonconverter(o):
  if isinstance(o, datetime.timedelta):
      return str(o)
