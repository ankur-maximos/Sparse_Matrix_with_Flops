#ifndef COO_H_
#define COO_H_
#include "tools/mm_io.h"
#include "CSR.h"

class COO {
private:
	int * cooRowIndex;
	int * cooColIndex;
	QValue * cooVal;
	int rows, cols, nnz;
public:
  COO(const char fname[]);
	COO();
	COO(const QValue* const cooVal, const int* const cooColIndex,
      const int* const cooRowIndex, const int rows, const int cols, const int nnz);
	//virtual ~COO();
  void dispose();
	void readMatrixMarketFile(const char fname[]);
/*Instead of read matrix read its transpose. It is useful for RMCL which process column instead of row.*/
  void readTransposedSNAPFile(const char fname[]);
  void addSelfLoopIfNeeded();
	void output(const char* msg);
	CSR toCSR() const;
  void makeOrdered() const;
};

#endif /* COO_H_ */
