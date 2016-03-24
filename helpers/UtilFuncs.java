package helpers;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class UtilFuncs {
	
	private static final double LOWER = -70;
	private static final double UPPER = 70;

	static double logistic(double x) {
		return 1/(1 + Math.exp(-x));
	}
	
	public static double logisDiff(double x) {
		double invExp = Math.exp(-x);
		double value = invExp/square(1 + invExp);
		//XXX: value can be NaN if the square(1 + invExp) is too large
		// in that case the value  actually converges 0
		// we guard against NaN by returning 0 instead
		if (!Double.isNaN(value)) {
			return value;
		} else {
			return 0;
		}
	}

	public static RealMatrix cutoff(RealMatrix matrix) {
		int rowDim = matrix.getRowDimension();
		int colDim = matrix.getColumnDimension();
		
		RealMatrix cutoffMatrix = new Array2DRowRealMatrix(rowDim, colDim);
		for (int i = 0; i < rowDim; i++) {
			for (int j = 0; j < colDim; j++) {
				double entry = matrix.getEntry(i, j);
				double logisticValue = logistic(entry);
				if (!Double.isNaN(logisticValue)) {
					cutoffMatrix.setEntry(i, j, logisticValue);
				} else {
					if (entry < LOWER) {
						cutoffMatrix.setEntry(i, j, 0);
					}
					if (entry > UPPER) {
						cutoffMatrix.setEntry(i, j, 1);
					}
				}
			}
		}
		
		return cutoffMatrix;
	}

	public static double square(double d) {
		return Math.pow(d, 2);
	}

	public static double sqFrobNorm(RealMatrix matrix) {
		
		return square(matrix.getFrobeniusNorm());
	}
}
