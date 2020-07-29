import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Properties;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * 
 * @author jason
 *
 */
public class SVM {

	public static void main(String[] args) throws Exception {
		InputStream input = new FileInputStream("infos.properties");
		Properties prop = new Properties();
		// load a properties file
		prop.load(input);
		
	    DataSource source = new DataSource(prop.getProperty("trainingFile"));
	    Instances train = source.getDataSet();
	    train.setClass(train.attribute(prop.getProperty("attribute")));  // target attribute: (outlook)
	      
	    //build model
	    SMO model=new SMO();
	    model.buildClassifier(train);
	    System.out.println("=== Classifier model (full training set) ===\n");
	    System.out.println(model);

	    //evaluation
	    DataSource testSource = new DataSource(prop.getProperty("evaluationFile"));
	    Instances test = testSource.getDataSet();
	    test.setClass(test.attribute(prop.getProperty("attribute")));
	    Evaluation eval = new Evaluation(test);
	    eval.evaluateModel(model,test);
	    System.out.println("=== Evaluation on test set ===");
	    System.out.println(eval.toSummaryString());
	}
}
