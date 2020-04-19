package com.ru8anraj.statj.linear;

import com.ru8anraj.statj.linear.beans.SimpleLinearBetas;
import com.ru8anraj.statj.linear.beans.SimpleLinearInitCalcData;

import java.util.ArrayList;
import java.util.List;

public class SimpleLinearRegression {

    private List<Long> X = new ArrayList<>(); // predictors
    private List<Long> y = new ArrayList<>(); // outcomes
    private SimpleLinearBetas beta = new SimpleLinearBetas(); // co-eff value for the prediction

    private SimpleLinearInitCalcData initCalc(List<Long> X, List<Long> y) {
        /*!
         * initCalc method
         * @params -> {List of Long} predictors - X
         * @params -> {List of Long} outcomes - y
         *
         * @purpose -> calculates the mean value of X and y
         *          -> calculates the numerator function and denominator function
         *
         * @returns -> initial calculation data
         *              { meanOfX, meanOfY, numerator, denominator }
         *
         */

        long sumOfX;
        long sumOfY;
        sumOfX = X.stream().mapToLong(x -> x).sum();
        sumOfY = y.stream().mapToLong(Y -> Y).sum();

        long meanOfX = sumOfX / X.size();
        long meanOfY = sumOfY / y.size();
        long numerator = 0;
        long denominator = 0;

        for (int i=0; i<X.size(); i++) {
            numerator += (X.get(i)-meanOfX) * (y.get(i)-meanOfY);
            denominator += (X.get(i)-meanOfX) * (X.get(i)-meanOfX);
        }

        SimpleLinearInitCalcData initCalcData = new SimpleLinearInitCalcData();
        initCalcData.setMeanOfX(meanOfX);
        initCalcData.setMeanOfY(meanOfY);
        initCalcData.setNumerator(numerator);
        initCalcData.setDenominator(denominator);
        return initCalcData;
    }

    private void SLR() {
        /*!
         * SLR method
         * @params -> none
         *
         * @purpose -> calls initial calculation function to pre-process the data
         *          -> sets the beta value
         *
         */
        SimpleLinearInitCalcData reqValues = initCalc(this.X, this.y);
        long slope = reqValues.getNumerator() / reqValues.getDenominator();
        long yIntercept = reqValues.getMeanOfY() - (slope * reqValues.getMeanOfX());
        beta.setSlope(slope);
        beta.setyIntercept(yIntercept);
    }

    public void fit(List<Long> X, List<Long> y) {
        /*!
         * FIT method
         * @params -> {List of Long} predictors - X
         * @params -> {List of Long} outcomes - y
         *
         * @purpose -> assigns predictors and outcomes to the X and y variable resp
         *          -> calls SLR method to calculate slope and y_intercept values (constants)
         *
         */
        if (X.size() != y.size())
            throw new RuntimeException("Input and Output training data should have same length");

        this.X = X;
        this.y = y;
        SLR();
    }

    public Long predict(long predictor) {
        /*!
         * PREDICT method
         * @params -> {Number} predictors
         *
         * @purpose -> predict the expected outcome by multiplying the new predictors with the constants
         *
         * @returns -> predicted value (y = mx + c)
         * i.e., (slope * predictor) + yIntercept
         */

        if(this.X.size() == 0 || this.y.size() == 0)
            throw new RuntimeException(">> Missing Fit: fit your dataset using fit() <<");

        return (beta.getSlope() * predictor) + beta.getyIntercept();
    }

    public SimpleLinearBetas coeff() {
        /*!
         * COEFF method
         * @params -> none
         *
         * @returns -> beta values
         */

        if (this.X.size() == 0 || this.y.size() == 0)
            throw new RuntimeException(">> MISSING FIT: fit your dataset using fit() <<");

        return this.beta;
    }
}
