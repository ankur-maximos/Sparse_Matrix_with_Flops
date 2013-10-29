#ifndef KEY_VALUE_QSORT_
#define KEY_VALUE_QSORT_

template <typename Key, typename Value>
void key_value_qsort (Key *keys, Value *values, int n) {
  if (n < 2)
    return;
  int p = keys[n >> 1];
  int *l = keys;
  int *r = keys + n - 1;
  while (l <= r) {
    if (*l < p) {
      l++;
    } else if (*r > p) {
      r--;
    } else {
      Value *vl = values + (l - keys);
      Value *vr = values + (r - keys);
      Value vt = *vl; *vl = *vr; *vr = vt;
      int t = *l; *l++ = *r; *r-- = t;
    }
  }
  key_value_qsort(keys, values, r - keys + 1);
  key_value_qsort(l, values + (l - keys), keys + n - l);
}
#endif
