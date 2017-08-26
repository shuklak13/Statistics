# Eigenvectors and Eigenvalues
* For a matrix A, **eigenvectors** are those vectors whose direction doesn't change when multiplied by that matrix; in other words, they become a scalar multiple of themselves.
* An **eigenvalue** is the scalar multiplier c that gives the same product with an eigenvector as the matrix A does. In other words, `Ax = cx`.
* **Eigen Decomposition Theorem**: a square matrix can always be decomposed into eigenvalues and eigenvectors
	* Every nxn matrix has n eigenvectors, and each eigenvector has a corresponding eigenvalue, but these eigenvectors need not be linearly independent

## Finding Eigenvalues
* Recall that for matrix A, eigenvector x, and eigenvalue c, `Ax=cx`.
* Therefore, `(A-cI)x=0`
* It is known that `Bx=0 => det(B)=0` (??? is this true).
* Therefore, `det(A-cI)=0`. This is the **Characteristic Equation**. (why's it called this?)
	* note that A-cI is just A with c subtracted from the diagonal
	* there is also a **Characteristic Polynomial**, which is `det(A-cI)=p(c)`.
* **To find the eigenvector c**, solve the Charcteristic Equation.

## Finding the Eigenvectors
* once you have the eigenvalue c, solve `(A-cI)x=0` to find the eigenvector x
	* you can solve the equation via Gaussian Elimination

# Properties of Matrices with Determinant 0
* The following are equivalent
	* The columns/rows are linearly dependent
	* The matrix is singular (not invertible)
	* A parallelipiped formed by the column/row vectors would have zero volume (because the vectors are linearly dependent, so they would just collapse on each other)
	* Any multiple of the matrix will also have determinant zero
	* The matrix cannot row-reduce to an identity matrix
	* The matrix's rank is not equal to its # of columns/rows
* Sources:
	* http://linear.ups.edu/html/section-PDM.html
	* https://math.stackexchange.com/questions/355644/what-does-it-mean-to-have-a-determinant-equal-to-zero
