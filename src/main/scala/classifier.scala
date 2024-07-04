package classifier

import scala.compiletime.ops.double
import util.Random

class LogisticRegression:
  private var weights: Array[Double] = Array[Double]()
  private var bias: Double = 0
  private var training = false
  
  def initialize(nbVariables: Int): Unit =
    this.weights = Array.fill(nbVariables){0}
    this.bias = 0
    this.training = false

  def predict(X: Array[Array[Double]]): Array[Double] =
    val predictions = X.map(xi => {
      // 1/(1 + exp(-w*xi-bias)), w is the weight vectior and b is the bias
      1/(1+ math.exp(-(this.weights.zip(xi).map((wj, xij) => wj * xij).sum() + this.bias)))
    })
    if this.training then 
      return predictions 
    else 
      return predictions.map(u => if u>=0.5 then 1 else 0)

  // def predict(x: Array[Double]): Double =
  //   return 1/(1+ math.exp(-(this.weights.zip(x).map((wj, xj) => wj * xj).sum() + this.bias)))

  // def gradient(x: Array[Double], yhat: Double, y: Double): (Array[Double], Double) =
  //   val d = yhat - y
  //   return (x.map(_ * d), d)

  def gradient(X: Array[Array[Double]], yhat: Array[Double], y: Array[Double]): (Array[Double], Double) =
    val batchSize = X.length
    // For each xi in the batch X, for each wj in the weight vector d_Loss/d_wj(xi) = di*xij
    // where di = yhati - yi, with yhati the prediction of the model on xi and yi the true label of xi
    val d = yhat.zip(y).map((u, v) => u - v)
    val batchGradients = X.zip(d).map((xi, di) => xi.map(_ * di))
    var gradients: Array[Double] = Array.fill(this.weights.length){0}
    batchGradients.foreach(a => {
      for j <- 0 to a.length - 1
      do gradients(j) += a(j)
    })
    return (gradients.map(_ / batchSize), d.sum()/batchSize)

  // Cross-entropy los function
  def loss(yhat: Array[Double], y: Array[Double]): Double =
    val batchSize = y.length
    val losses = yhat.zip(y).map((yhati, yi) => -yi*math.log(yhati) - (1-yi)*math.log(1.0-yhati))
    return losses.sum()/batchSize

  def updateParams(paramsGrad: (Array[Double], Double), lr:Double): Unit =
    val (weightsGrad, biasGrad) = paramsGrad
    this.bias -= lr * biasGrad
    for j <- Range(0, weightsGrad.length)
    do this.weights(j) -= lr * weightsGrad(j)


  def train(X: Array[Array[Double]], y: Array[Double], lr: Double = 1e-3, maxIter: Int = 100, tol: Double = 1e-4, seed: Int = 0): Unit =
    this.training = true
    val nbSamples: Int = X.length
    val nbVariables: Int = X(0).length
    initialize(nbVariables)
    val generator = Random(seed)
    var gradNorm: Double = 1
    var iter = 1
    while gradNorm > tol && iter <= maxIter do
      var totalLoss: Double = 0
      for i <- generator.shuffle(Range(0, nbSamples))
      do
        val yhat = predict(Array(X(i)))
        val loss = this.loss(yhat, Array(y(i)))
        val grad = gradient(Array(X(i)), Array(yhat(0)), Array(y(i)))
        updateParams(grad, lr)
        gradNorm = math.sqrt(grad(0).map(g => g*g).sum() + grad(1)*grad(1))
        totalLoss += loss
      totalLoss /= nbSamples
      printf("Iter: %d, loss: %f", iter, totalLoss)
      println()
      iter += 1
    this.training = false
