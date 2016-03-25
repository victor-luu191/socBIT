package core;

import org.apache.commons.math3.linear.RealMatrix;

class ErrorCal {
	
	// Pre: both weight matrices are square with the same dimension
	static RealMatrix edgeWeightErrors(RealMatrix est_weights, RealMatrix obs_weights) {
		// Ad-hoc trick: as w_{u, u}'s do NOT exist, we need to exclude errors due to estimating them by 
		// resetting them equal to estimated weights
		int numUser = obs_weights.getColumnDimension();
		RealMatrix imputed_weights = obs_weights.copy();	// safe-guard copy
		for (int u = 0; u < numUser; u++) {
			imputed_weights.setEntry(u, u, est_weights.getEntry(u, u));	  
		}
		RealMatrix edge_weight_errors = est_weights.subtract(imputed_weights);
		return edge_weight_errors;
	}
	
	static RealMatrix ratingErrors(RealMatrix est_ratings, RealMatrix obs_ratings) {
		// XXX: many ratings r_{u,i} are missing as u may not rate i. Again we should exclude the errors from these missing ratings by  
		// similar trick i.e. force the missing ratings equal to estimated values (so that the errors vanish)  
		RealMatrix imputed_ratings = imputeNAs(obs_ratings, est_ratings);
		RealMatrix rating_errors = est_ratings.subtract(imputed_ratings);
		return rating_errors;
	}
	
	/**
	 * NAs in {@link mat} are marked by some invalid value i.e. null, 
	 * in the case of rating, we used -1 as marker 
	 */
	static RealMatrix imputeNAs(RealMatrix mat, RealMatrix estimated_values) {
		
		RealMatrix imputed_mat = mat.copy();	 // safe-guard copy
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				if (imputed_mat.getEntry(i, j) == -1) {
					imputed_mat.setEntry(i, j, estimated_values.getEntry(i, j));
				}
			}
		}
		return imputed_mat;
	}
}
