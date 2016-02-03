package core;

import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Parameters {
	
	/**
	 * represent decision preference of users i.e. whether a user prefers brand-based or topic-based adopts
	 * Range: [0,1]; > 0.5 if the user prefers topic-based, < 0.5 if  the user prefers brand-based
	 */
	double[] userDecisionPrefs;	
	
	RealMatrix topicUser;	// represent topic interests of users, later need to normalize for each user
	RealMatrix brandUser;	// represent brand interests of users, later need to normalize for each user
	
	RealMatrix topicItem;	// represent relevance of each item with different topics, later need to normalize for each topic

	// represent popularity of each item under each brand producing the item; 
	// if a brand b does not produce an item i then the entry (i,b) is simply 0 
	RealMatrix brandItem;	
	
	public Parameters(int numUser, int numItem, int numTopic, int numBrand) {
		
		topicUser = new Array2DRowRealMatrix(numTopic, numUser);
		brandUser = new Array2DRowRealMatrix(numBrand, numUser);
		
		userDecisionPrefs = new double[numUser];
		// as we expect that most users are neutral, neither brand-based nor topic-based extreme, 
		// we initialize all decision prefs as 0.5
		Arrays.fill(userDecisionPrefs, 0.5);	
		
		topicItem = new Array2DRowRealMatrix(numTopic, numItem);
		brandItem = new Array2DRowRealMatrix(numBrand, numItem);
	}
}
