package core;

import org.apache.commons.math3.linear.*;

class SocBIT_Estimator {
	
	private SocBIT_Params params;
	
	/**
	 * fields derived from {@code params}
	 */
	private int numUser;
	private DiagonalMatrix decisionPrefs;
	private RealMatrix idMat;

	public SocBIT_Estimator(SocBIT_Params params) {
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
