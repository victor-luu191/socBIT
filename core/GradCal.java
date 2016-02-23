package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.InvalidModelException;

public abstract class GradCal {
	
	protected Hypers hypers;
	// derived fields
	protected int numTopic;
	Dataset ds;
	protected RealMatrix estimated_ratings;
	protected RealMatrix estimated_weights;

	public GradCal(Trainer trainer) {
		
		numTopic = trainer.numTopic;
		ds = trainer.ds;
		hypers = trainer.hypers;
		
	}
	
	// model is the trainer's model
	abstract Params calculate(Params params);
	
	/**
	 * Wrapper for calculations of gradient of item topic feats
	 * @param itemIndex
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr: errors of estimating ratings of the item
	 * @return
	 */
	abstract RealVector itemTopicGrad(Params params, int itemIndex, RealMatrix rating_errors);

	/**
	 * @param u
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr
	 * @param weightLambda 
	 * @param strengthErr: errors in estimating strength of relationships of the user 
	 * @return
	 */
	RealVector userTopicGrad(Params params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
		RealVector topicGrad = new ArrayRealVector(numTopic);
		if (params instanceof SocBIT_Params) {
			SocBIT_Params tmp_params = new SocBIT_Params((SocBIT_Params) params);
			topicGrad = socBIT_userTopicGrad(tmp_params, u, rating_errors, edge_weight_errors);
		} 
		else {// simpler parameters with only topic features
			topicGrad = ste_userTopicGrad(params, u, rating_errors);
		}
		
		return topicGrad;
	}
	
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
