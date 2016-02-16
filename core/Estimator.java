package core;

import org.apache.commons.math3.linear.*;

class Estimator {
	
	private Parameters params;
	
	/**
	 * fields derived from {@code params}
	 */
	private int numUser;
	private DiagonalMatrix decisionPrefs;
	private RealMatrix idMat;

	public Estimator(Parameters params) {
		this.params = params;
		decisionPrefs = new DiagonalMatrix(params.userDecisionPrefs);
		numUser = params.brandUser.getColumnDimension();
		idMat = MatrixUtils.createRealIdentityMatrix(numUser);
	}
	
	RealMatrix estRatings() {
		
		RealMatrix topicRatings = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicItem);
		RealMatrix brandRatings = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandItem);
		RealMatrix est_ratings =  topicRatings.add(brandRatings); 
		return est_ratings;
	}
	
	RealMatrix estWeights() {
		
		RealMatrix topicWeights = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicUser);
		RealMatrix brandWeights = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandUser);
		RealMatrix est_edge_weights = topicWeights.add(brandWeights);
		return est_edge_weights;
	}
	
	RealMatrix bound(RealMatrix matrix) {
		return logisticMat(matrix);
	}
	
	private double logistic(double x) {
		return 1/(1 + Math.exp(-x));
	}
	
	private RealMatrix logisticMat(RealMatrix matrix) {
		
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
}
