package core;

import org.apache.commons.math3.linear.RealMatrix;

import defs.Dataset;
import defs.Hypers;

public class GD_Trainer {
	
	int numTopic;
	Dataset ds;
	Hypers hypers;
	
	/**
	 * 
	 * @param initParams
	 * @param resDir
	 * @return local optimal parameters which give a local minimum of the objective function (minimum errors + regularizers)
	 */
	private Parameters gradDescent(Parameters initParams, String resDir) {
		// TODO Auto-generated method stub
		
		GradCalculator gradCal = new GradCalculator(this);
		
		Parameters cParams = new Parameters(initParams);
		double cValue = objValue(initParams);
		double difference = 100.0;
	}

	private double objValue(Parameters params) {
		
		Estimator estimator = new Estimator(params);
		RealMatrix estimated_ratings = estimator.estRatings();
		RealMatrix estimated_weights = estimator.estWeights();
		
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(estimated_weights, ds.edge_weights);
		RealMatrix rating_errors = ErrorCal.ratingErrors(estimated_ratings, ds.ratings);
		
		double val = square(rating_errors.getFrobeniusNorm());
		val += hypers.weightLambda * square(edge_weight_errors.getFrobeniusNorm());
		val += hypers.topicLambda * ( square(params.topicUser.getFrobeniusNorm()) + square(params.topicItem.getFrobeniusNorm()) );
		val += hypers.brandLambda * ( square(params.brandUser.getFrobeniusNorm()) + square(params.brandItem.getFrobeniusNorm()) );
		for (int u = 0; u < ds.numUser; u++) {
			val += hypers.decisionLambda * square(params.userDecisionPrefs[u] - 0.5);
		}
		
		return val;
	}
	
	double square(double d) {
		return Math.pow(d, 2);
	}
}
