package core;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import defs.Dataset;
import defs.Params;

public abstract class RecSysCal {
	
	protected RealMatrix estimated_ratings;
	
	public RecSysCal(Dataset ds) {
		estimated_ratings = new Array2DRowRealMatrix(ds.numUser, ds.numItem);
	}
	
	abstract double objValue(Params params);
	
	abstract void estRatings(Params params);
	
	abstract RealMatrix calRatingErrors(Params params);
}
