package core;

import org.apache.commons.math3.linear.RealMatrix;

class ErrorCal {
	
	// Pre: both weight matrices are square with the same dimension
	static RealMatrix edgeWeightErrors(RealMatrix estimated_weights, RealMatrix actual_edge_weights) {
		// as w_{u, u}'s do NOT exist, we need to exclude errors due to estimating them by the following trick 
		int numUser = actual_edge_weights.getColumnDimension();
		for (int u = 0; u < numUser; u++) {
			actual_edge_weights.setEntry(u, u, estimated_weights.getEntry(u, u));	  
		}
		RealMatrix edge_weight_errors = actual_edge_weights.subtract(estimated_weights);
		return edge_weight_errors;
	}
	
	static RealMatrix ratingErrors(RealMatrix estimated_ratings, RealMatrix actual_ratings) {
		// XXX: many ratings r_{u,i} are missing as u may not rate i. Again we should exclude the errors from these missing ratings by  
		// similar trick i.e. force the missing ratings equal to estimated values (so that the errors vanish)  
		actual_ratings = fillNAs(actual_ratings, estimated_ratings);
		RealMatrix rating_errors = actual_ratings.subtract(estimated_ratings);
		return rating_errors;
	}
	
	/**
	 * NAs in {@link mat} are marked by some invalid value i.e. null, 
	 * in the case of rating, we use -1 as marker 
	 */
	static RealMatrix fillNAs(RealMatrix mat, RealMatrix estimated_values) {
		// TODO Auto-generated method stub
		RealMatrix filled_mat = mat;
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				if (filled_mat.getEntry(i, j) == -1) {
					filled_mat.setEntry(i, j, estimated_values.getEntry(i, j));
				}
			}
		}
		return filled_mat;
	}
}
