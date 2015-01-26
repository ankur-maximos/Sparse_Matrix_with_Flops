#include "CSR.h"
#include "COO.h"
#include "tools/util.h"
#include <vector>

using namespace std;

void CSR_PM_Test() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR M = coo.toCSR();
  coo.dispose();
  M.output("\nM");
  int P[] = {1, 3, 0, 2};
  CSR pM = M.PM(P);
  pM.output("\nPM");
  int prowPtr[] = {0, 1, 2, 3, 4};
  int pcolInd[] = {1, 3, 0, 2};
  double pValues[] = {1.0, 1.0, 1.0, 1.0};
  COO cooP(pValues, pcolInd, prowPtr, 4, 4, 4);
  CSR csrP = cooP.toCSR();
  CSR mPM = csrP.spmm(M);
  M.dispose();
  mPM.makeOrdered();
  pM.makeOrdered();
  bool isSame = mPM.isEqual(pM);
  mPM.output("mPM");
  pM.dispose();
  assert(isSame == true);
  printf("%s Passed\n", __func__);
}

void CSR_MP_Test() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR M = coo.toCSR();
  coo.dispose();
  int P[] = {1, 3, 0, 2};
  CSR mP = M.MP(P);
#ifdef DEBUG
  M.output("\nM");
  mP.output("\nMP");
#endif
  int prowPtr[] = {0, 1, 2, 3, 4};
  int pcolInd[] = {1, 3, 0, 2};
  double pValues[] = {1.0, 1.0, 1.0, 1.0};
  COO cooP(pValues, pcolInd, prowPtr, 4, 4, 4);
  CSR csrP = cooP.toCSR();
  CSR mMP = M.spmm(csrP);
  M.dispose();
  mMP.makeOrdered();
  mP.makeOrdered();
  bool isSame = mMP.isEqual(mP);
#ifdef DEBUG
  mMP.output("mMP");
#endif
  mP.dispose();
  assert(isSame == true);
  printf("%s Passed\n", __func__);
}

void CSR_initWithDenseMatrixTest() {
  const QValue dMM[] = {
    1, 2, 0, 0, 0, 0,
    0, 0, 3, 0, 0, 0,
    0, 4, 0, 0, 0, 5,
    0, 0, 0, 0, 2, 0,
    3, 0, 0, 1, 0, 8
  };
  const int rows = 5, cols = 6;
  CSR A;
  A.initWithDenseMatrix(dMM, rows, cols);
  const int rowPtr[] = {0, 2, 3, 5, 6, 9};
  const int colInd[] = {0, 1, 2, 1, 5, 4, 0, 3, 5};
  const QValue values[] = {1, 2, 3, 4, 5, 2, 3, 1, 8};
  for (int i = 0; i < rows + 1; ++i) {
    assert(rowPtr[i] == A.rowPtr[i]);
  }
  for (int j = 0; j < A.nnz; ++j) {
    assert(colInd[j] == A.colInd[j]);
    assert(fabs(values[j] - A.values[j]) < 1e-8);
  }
  printf("%s Passed\n", __func__);
}

int partition(int* a,int *b,int p,int r) {
int x = a[r];
int j = p-1;
int temp;
for(int i=p;i<=r-1;i++) {
if(a[i]<=x) {
j++;
temp = a[j];
a[j] = a[i];
a[i] = temp;
temp = b[j];
b[j] = b[i];
b[i] = temp;
}
}
temp = a[j+1];
a[j+1] = a[r];
a[r] = temp;
temp = b[j+1];
b[j+1] = b[r];
b[r] = temp;
return j+1;
}

//Sorting of two arrays
//key->a , value->b
void quicksort(int *a,int *b,int p,int r) {
if(p>=r) {return;}
int m = partition(a,b,p,r);
//printf("%d ",m);
quicksort(a,b,p,m-1);
quicksort(a,b,m+1,r);
}

void print_array(int *arr,int size) {
printf ("print_array \n");
//print_array(values_arr,rows_b);
for(int i=0;i<size;i++) {
printf("%d ",arr[i]);
}
printf("\n");
}


// Here bin_limits contains num values and we can have (num+1) bins from them so i am returning a vector which will have (num+1) values
vector<int> filter_rows(int limit,CSR a,CSR b,vector<int> bin_limits) {
vector<int> filter_arr;
int *rowptr_b = b.rowPtr;
int *rowptr_a = a.rowPtr;
int *colptr_a = a.colInd;
int nnz_b  = b.nnz;
int rows_b = b.rows;
int rows_a = a.rows;
int cols = a.cols;
int *values_arr = (int*)calloc(rows_b,sizeof(int));
int *count_arr = (int*)calloc(rows_b,sizeof(int));
int diff = 0;
//setting total values of non-zero values for B
for(int i=1;i<=rows_b;i++) {
diff = rowptr_b[i] - rowptr_b[i-1];
values_arr[i-1] = diff;
}

//computing summation of(values_arr) for those rows with non-zero values greater than limit
for(int i=1;i<=rows_a;i++) {
diff = rowptr_a[i]-rowptr_a[i-1];
if(diff>=limit) {
for (int j=rowptr_a[i-1];j<rowptr_a[i];j++) {
//column value for which the count have to be increased
int col_val = colptr_a[j];
count_arr[col_val]++;
}
}
}
//print_array(values_arr,rows_b);
quicksort(values_arr,count_arr,0,rows_b-1);
//print_array(values_arr,rows_b);
//print_array(count_arr,rows_b);
//computing the bins
vector<int>::iterator it;
int pos = 0;
for(it=bin_limits.begin();it<bin_limits.end();it++) {
diff = 0;
for(;pos<rows_b;) {
if(values_arr[pos]<=*it) {
diff+=count_arr[pos];
pos++;
}
else {break;}
}
filter_arr.push_back(diff);
}
diff = 0;
for(;pos<rows_b;pos++) {
diff+=count_arr[pos];
}
filter_arr.push_back(diff); 
return filter_arr;
}

void print_vector(vector<int> a) {
vector<int>::iterator it;
for (it=a.begin();it<a.end();++it) {
printf("%d\t",*it);
}
printf("\n");
}

void print_bounds (vector<int> a) {
vector<int>::iterator it;
for (it=a.begin();it<a.end();++it) {
printf("<=%d\t",*it);
}
printf(">%d\n",a.at(a.size()-1));
}

int main() {
  int rows = 4, cols = 6, nnz = 7;
  const int rowIndex[] = {0, 0, 1, 2, 2, 3, 3};
  const int colIndex[] = {1, 4, 2, 0, 5, 1, 3};
  const double values[] = {2.0, 6.0, 3.0, 4.0, 7.0, 1.0, 5.0};

  int rows_b = 6, cols_b = 5, nnz_b = 9;
  const int rowIndex_b[] = {1, 1, 3, 3, 3, 4, 4, 5, 5};
  const int colIndex_b[] = {2, 4, 1, 3, 4, 2, 4, 0, 2};
  const double values_b[] = {2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 7.0, 8.0, 9.0};

  int limit = 2;
  static const int arr[] = {0,1,2,3};
  vector<int> bin_bounds(arr,arr+4);
  //print_vector(bin_bounds);

  COO coo_a(values, colIndex, rowIndex, rows, cols, nnz);
  COO coo_b(values_b, colIndex_b, rowIndex_b, rows_b, cols_b, nnz_b);
  CSR a = coo_a.toCSR();
  CSR b = coo_b.toCSR();
  //csr.averAndNormRowValue();
  a.output("csr");
  b.output("csr");
  /*int* temp = a.rowPtr;
  for(int i=0;i<=4;i++) {
    printf("%d ",temp[i]);
  }
  temp = b.rowPtr;
  printf("\n");
  for(int i=0;i<=6;i++) {
    printf("%d ",temp[i]);
  }*/

  // printing the bin bounds
  print_bounds(bin_bounds);  
  vector<int> bins = filter_rows(limit,a,b,bin_bounds);
  print_vector(bins);
  return 0;
}
