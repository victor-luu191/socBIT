package core;

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
		
		return null;
	}

	@Override
	RealVector calItemTopicGrad(Params params, int itemIndex) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	RealVector calUserTopicGrad(Params params, int u) {
		// TODO Auto-generated method stub
		return null;
	}

}
