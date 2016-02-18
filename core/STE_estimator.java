package core;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

class STE_estimator {
	
	private Params params;
	private int numUser;
	
	/**
	 * Estimated rating r_{u,i} by STE (social trust ensemble) model in Hao Ma paper: Learning to Recommend with Social Trust Ensemble
	 * @param u
	 * @param i
	 * @param edge_weights
	 * @param alpha: control how much each user trusts himself vs. trust friends
	 * @return
	 */
	double estRatings(int u, int i, RealMatrix edge_weights, double alpha) {
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
}
