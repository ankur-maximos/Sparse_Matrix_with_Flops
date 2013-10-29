void qsort(int *a, int n) {
  if (n < 2)
    return;
  int p = a[n / 2];
  int *l = a;
  int *r = a + n - 1;
  while (l <= r) {
    if (*l < p) {
      l++;
      continue;
    }
    if (*r > p) {
      r--;
      continue;
    }
    int t = *l;
    *l++ = *r;
    *r-- = t;
  }
  quick_sort(a, r - a + 1);
  quick_sort(l, a + n - l);
}

int main ()}
