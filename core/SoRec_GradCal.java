package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Params;
import defs.SoRecParams;

public class SoRec_GradCal extends GradCal {

	private SoRec_Cal calculator;
	private RealMatrix estimated_weights;
	private RealMatrix edge_weight_errors;
	
	@Override
	Params calculate(Params params) {
		// TODO Auto-generated method stub
		SoRecParams soRecParams = (SoRecParams) params;
		estimated_ratings = calculator.estRatings(soRecParams);
		rating_errors = calculator.calRatingErrors(soRecParams);
		estimated_weights = calculator.estWeights(soRecParams);
		edge_weight_errors = calculator.calEdgeWeightErrors(soRecParams);
		
		SoRecParams grad = new SoRecParams(ds.numUser, ds.numItem, this.numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, calItemTopicGrad(params, i));
		}
		
		for (int u = 0; u < ds.numUser; u++) {
			grad.topicUser.setColumnVector(u, calUserTopicGrad(params, u));
			grad.zMatrix.setColumnVector(u, calZGrad(soRecParams, u));
		}
		
		return grad;
	}

	private RealVector calZGrad(SoRecParams params, int u) {
		
		RealVector zVector = params.zMatrix.getColumnVector(u);
		RealVector zGrad = zVector.mapMultiply(hypers.topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int v = 0; v < ds.numUser; v++) {
			double trustErr = edge_weight_errors.getEntry(v, u);
			if (trustErr != 0) {
				RealVector vTopicFeats = params.topicUser.getColumnVector(v);
				double estTrust = estimated_weights.getEntry(v, u);
				double logisDiff = UtilFuncs.logisDiff(estTrust);
				RealVector modifiedFeats = vTopicFeats.mapMultiply(trustErr*logisDiff);
				sum = sum.add(modifiedFeats);
			}
		}
		
		zGrad = zGrad.add(sum.mapMultiply(hypers.weightLambda));
		return zGrad;
	}

	@Override
	RealVector calItemTopicGrad(Params params, int itemIndex) {
		// TODO Auto-generated method stub
		double topicLambda = hypers.topicLambda;
		RealVector itemTopicFeats = params.topicItem.getColumnVector(itemIndex);
		RealVector topicGrad = itemTopicFeats.mapMultiply(topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double rating_err = rating_errors.getEntry(u, itemIndex);
			if (rating_err != 0) {
				RealVector userTopicFeat = params.topicUser.getColumnVector(u);
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, itemIndex));
				sum = sum.add(userTopicFeat.mapMultiply(rating_err*logisDiff));
			}
		}
		
		topicGrad = topicGrad.add(sum);
		return topicGrad;
	}

	@Override
	RealVector calUserTopicGrad(Params params, int u) {
		
		RealVector topicFeats = params.topicUser.getColumnVector(u);
		RealVector userTopicGrad = topicFeats.mapMultiply(hypers.topicLambda);
		
		RealVector rating_sum = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			double rError = rating_errors.getEntry(u, i);
			if (rError != 0) {
				RealVector curItemTopicFeat = params.topicItem.getColumnVector(i);
				double ratingLogisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
				RealVector modified_topicFeat = curItemTopicFeat.mapMultiply(rError*ratingLogisDiff);
				rating_sum = rating_sum.add(modified_topicFeat);
			}
		}
		
		SoRecParams soRecParams = (SoRecParams) params;
		RealVector edge_weight_sum = new ArrayRealVector(numTopic);
		for (int v = 0; v < ds.numUser; v++) {
			double trustErr = edge_weight_errors.getEntry(u, v);
			if (trustErr != 0) {
				RealVector zOfFriend = soRecParams.zMatrix.getColumnVector(v);
				double weightLogisDiff = UtilFuncs.logisDiff(estimated_weights.getEntry(u, v));
				RealVector modified_topicFeat = zOfFriend.mapMultiply(trustErr*weightLogisDiff);
				edge_weight_sum = edge_weight_sum.add(modified_topicFeat);
			}
		}
		
		userTopicGrad = userTopicGrad.add(rating_sum).add(edge_weight_sum.mapMultiply(hypers.weightLambda));
		return userTopicGrad;
	}

}
