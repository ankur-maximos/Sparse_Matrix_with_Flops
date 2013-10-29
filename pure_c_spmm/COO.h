#ifndef COO_H_
#define COO_H_
#include "mm_io.h"
#include "CSR.h"

class COO {
private:
	int * cooRowIndex;
	int * cooColIndex;
	double * cooVal;
	int rows, cols, nnz;
public:
  COO(const char fname[]);
	COO();
	virtual ~COO();
	void readMatrixMarketFile(const char fname[]);
	void output(const char* msg);
	CSR toCSR() const;
  void makeOrdered() const;
};

#endif /* COO_H_ */
