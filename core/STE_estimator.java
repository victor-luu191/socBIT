package core;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

class STE_estimator {
	
	private Params params;
	private int numUser;
	private double alpha;	// control how much each user trusts himself vs. trust friends
	private int numItem;
	private RealMatrix edge_weights;
	
	STE_estimator(Params params, double alpha, RealMatrix edge_weights) {
		this.params = params;
		this.alpha = alpha;
		numUser = params.topicUser.getColumnDimension();
		numItem = params.topicItem.getColumnDimension();
		this.edge_weights = edge_weights;
	}

	
	/**
	 * Estimated rating r_{u,i} by STE (social trust ensemble) model in Hao Ma paper: Learning to Recommend with Social Trust Ensemble
	 * @param u
	 * @param i
	 * @param edge_weights
	 * @return
	 */
	double estOneRating(int u, int i, RealMatrix edge_weights) {
		
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

	RealMatrix estRatings() {
		RealMatrix estimated_ratings = new Array2DRowRealMatrix(numUser, numItem);
		for (int u = 0; u < numUser; u++) {
			for (int i = 0; i < numItem; i++) {
				estimated_ratings.setEntry(u, i, estOneRating(u, i, edge_weights));
			}
		}
		return estimated_ratings;
	}
}
