package core;

import org.apache.commons.math3.linear.*;

class Estimator {
	
	private Parameters params;
	
	/**
	 * fields derived from {@code params}
	 */
	private DiagonalMatrix decisionPrefs;
	private int numUser;
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
}
