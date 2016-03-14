package core;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.Params;

public abstract class GradCal {
	
	protected Hypers hypers;
	protected int numTopic;
	Dataset ds;

	// derived fields
	protected RealMatrix estimated_ratings;
	protected RealMatrix rating_errors;

	// model is the trainer's model
	abstract Params calculate(Params params);
	
	abstract RealVector calItemTopicGrad(Params params, int itemIndex);	// RealMatrix rating_errors

	abstract RealVector calUserTopicGrad(Params params, int u);	// RealMatrix rating_errors, RealMatrix edge_weight_errors
	
	/**
	 * NAs in {@link mat} are marked by some invalid value i.e. null, 
	 * in the case of rating, we use -1 as marker 
	 */
	RealMatrix fillNAs(RealMatrix mat, RealMatrix estimated_values) {
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
