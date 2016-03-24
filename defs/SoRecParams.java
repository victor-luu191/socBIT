package defs;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class SoRecParams extends Params {

	public RealMatrix zMatrix;
	
	public SoRecParams(int numUser, int numItem, int numTopic) {
		super(numUser, numItem, numTopic);
		initUserTopicFeats(numUser, numTopic);
		initItemTopicFeats(numItem, numTopic);
		
		initUnifZ(numUser, numTopic);
	}

	private void initUnifZ(int numUser, int numTopic) {
		zMatrix = new Array2DRowRealMatrix(numTopic, numUser);
		RealVector uniformVector = uniformVector(numTopic);
		for (int u = 0; u < numUser; u++) {
			zMatrix.setColumnVector(u, uniformVector);
		}
	}

	public SoRecParams(SoRecParams params) {
		super(params);
		this.zMatrix = params.zMatrix;
	}

	public double sqDiff(SoRecParams other) {
		double topicDiff = topicDiff(other);
		double zDiff = UtilFuncs.sqFrobNorm(this.zMatrix.subtract(other.zMatrix));
		return topicDiff + zDiff;
	}

}
