package helpers;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class UtilFuncs {
	
	static double logistic(double x) {
		return 1/(1 + Math.exp(-x));
	}
	
	public static double logisDiff(double x) {
		double invExp = Math.exp(-x);
		return invExp/Math.pow(1 + invExp, 2);
	}

	static RealMatrix logisticMat(RealMatrix matrix) {
		int rowDim = matrix.getRowDimension();
		int colDim = matrix.getColumnDimension();
		RealMatrix logisMatrix = new Array2DRowRealMatrix(rowDim, colDim);
		for (int i = 0; i < rowDim; i++) {
			for (int j = 0; j < colDim; j++) {
				logisMatrix.setEntry(i, j, logistic(matrix.getEntry(i, j)));
			}
		}
		
		return logisMatrix;
	}
	
	public static RealMatrix bound(RealMatrix matrix) {
		return logisticMat(matrix);
	}
}
