package core;

import org.apache.commons.math3.linear.RealMatrix;

import defs.Params;

public abstract class RecSysCal {
	
	abstract double objValue(Params params);
	
	abstract RealMatrix estRatings(Params params);
	
	abstract RealMatrix calRatingErrors(Params params);
}
