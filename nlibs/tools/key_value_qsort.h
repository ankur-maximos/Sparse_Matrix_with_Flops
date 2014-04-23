#ifndef KEY_VALUE_QSORT_
#define KEY_VALUE_QSORT_

template <typename sKey>
bool lessThanFunction(const sKey &a, const sKey &b) {
  return a < b;
}

template <typename sKey, typename sValue>
void key_value_qsort (sKey *keys, sValue *values, long n,
    bool (*lessThan)(const sKey &a, const sKey &b)) {
  if (n < 2)
    return;
  int p = keys[n >> 1];
  sKey *l = keys;
  sKey *r = keys + n - 1;
  while (l <= r) {
    if (lessThan(*l, p)) {
      l++;
    } else if (lessThan(p, *r)) {
      r--;
    } else {
      sValue *vl = values + (l - keys);
      sValue *vr = values + (r - keys);
      sValue vt = *vl; *vl = *vr; *vr = vt;
      int t = *l; *l++ = *r; *r-- = t;
    }
  }
  key_value_qsort<sKey, sValue>(keys, values, r - keys + 1, lessThan);
  key_value_qsort<sKey, sValue>(l, values + (l - keys), keys + n - l, lessThan);
}

template <typename sKey, typename sValue>
void key_value_qsort(sKey *keys, sValue *values, long n) {
  bool (*functionPointer)(const sKey &, const sKey &) = &(lessThanFunction<sKey>);
  key_value_qsort<sKey, sValue>(keys, values, n, functionPointer);
}
#endif
