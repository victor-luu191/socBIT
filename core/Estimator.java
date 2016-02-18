package core;

import org.apache.commons.math3.linear.*;

class Estimator {
	
	private SocBIT_Params params;
	
	/**
	 * fields derived from {@code params}
	 */
	private int numUser;
	private DiagonalMatrix decisionPrefs;
	private RealMatrix idMat;

	public Estimator(SocBIT_Params params) {
		this.params = params;
		decisionPrefs = new DiagonalMatrix(params.userDecisionPrefs);
		numUser = params.brandUser.getColumnDimension();
		idMat = MatrixUtils.createRealIdentityMatrix(numUser);
	}
	
	RealMatrix socBIT_Ratings() {
		
		RealMatrix topicRatings = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicItem);
		RealMatrix brandRatings = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandItem);
		RealMatrix est_ratings =  topicRatings.add(brandRatings); 
		return est_ratings;
	}
	
	RealMatrix socBIT_Weights() {
		
		RealMatrix topicWeights = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicUser);
		RealMatrix brandWeights = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandUser);
		RealMatrix est_edge_weights = topicWeights.add(brandWeights);
		return est_edge_weights;
	}
	
	/**
	 * Estimated rating r_{u,i} by STE (social trust ensemble) model in Hao Ma paper: Learning to Recommend with Social Trust Ensemble
	 * @param u
	 * @param i
	 * @param edge_weights
	 * @param alpha: control how much each user trusts himself vs. trust friends
	 * @return
	 */
	double ste_Ratings(int u, int i, RealMatrix edge_weights, double alpha) {
		RealVector userTopicFeat = params.topicUser.getColumnVector(u);
		RealVector itemTopicFeat = params.topicItem.getColumnVector(i);
		double personal_rating = userTopicFeat.dotProduct(itemTopicFeat);
		
		double neighbor_rating = 0;
		for (int v = 0; v < numUser; v++)  {
			RealVector v_topicFeat = params.topicUser.getColumnVector(v);
			neighbor_rating += edge_weights.getEntry(v, u) * v_topicFeat.dotProduct(itemTopicFeat);
		}
		return alpha*personal_rating + (1 - alpha)*neighbor_rating;
	}
	
	RealMatrix soRecRatings() {
		RealMatrix estRatings = params.topicUser.transpose().multiply(params.topicItem);
		return estRatings;
	}
	
//	RealMatrix soRecEdgeWeights() {
//		
//	}
	
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
