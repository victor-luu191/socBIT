package defs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class SoRecParams extends Params {

	public RealMatrix zMatrix;
	
	public SoRecParams(int numUser, int numItem, int numTopic) {
		super(numUser, numItem, numTopic);
		zMatrix = new Array2DRowRealMatrix(numTopic, numUser);
	}

	public SoRecParams(SoRecParams params) {
		super(params);
		this.zMatrix = params.zMatrix;
	}

}
