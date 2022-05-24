// import * as tf from '@tensorflow/tfjs'
import * as tf from '@tensorflow/tfjs-node'
import file from "./file.js";
import Papa from "papaparse";


const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

const numIterations = 500;
const learningRate = 0.25;
const optimiser = tf.train.adam(learningRate);

var xs = [];
var ys = [];
var region = "US";
var sku = "256GB";
var degree = 1

var l_max = tf.tensor([1]);
var l_min = tf.tensor([1]);
var i_max = tf.tensor([1]);
var i_min = tf.tensor([1]);

const parseCSV = async (res) => {
    Papa.parse(file().training, {
        header: true,
        complete: res => {
            var prop = listAllProperties(res.data[0])[0]
            let propertyName = prop
            var res = res.data.filter(item => item.Model == sku && item[propertyName] == region)
            console.log('filtering:' + region + "/" + sku)
            xs = []
            ys = []
            res.forEach((item, index) => {
                xs.push(parseFloat(item['Reserve Time']))
                ys.push(parseFloat(item['Order Date']))
            })
        }
    });

}

function normalizeData() {
    const xs_ = tf.tensor2d(xs, [xs.length, 1])
    const ys_ = tf.tensor2d(ys, [ys.length, 1])
    const x_min = xs_.min();
    const x_max = xs_.max();
    const y_min = ys_.min();
    const y_max = ys_.max();

    xs = xs_.sub(x_min).div(x_max.sub(x_min))
    ys = ys_.sub(y_min).div(y_max.sub(y_min))

    l_max = y_max;
    l_min = y_min;
    i_max = x_max;
    i_min = x_min

}

function listAllProperties(o) {
    let objectToInspect = o;
    let result = [];

    while (objectToInspect !== null) {
        result = result.concat(Object.getOwnPropertyNames(objectToInspect));
        objectToInspect = Object.getPrototypeOf(objectToInspect)
    }

    return result;
}

function predict(x, opt) {
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
    // y = a * X ^ 2 + b * X + c
    // y = a * X + b
    return tf.tidy(() => {
        if (opt == 1) {
            return a.mul(x).add(b)
        }
        if (opt == 2) {
            return a.mul(x.pow(tf.scalar(2, 'int32'))).add(b.mul(x)).add(c)
        }
        if (opt == 3) {
            return a.mul(x.pow(tf.scalar(3, 'int32')))
                .add(b.mul(x.square()))
                .add(c.mul(x))
                .add(d)
        }

    });
}

function loss(prediction, labels) {
    const error = prediction.sub(labels).square().mean()
    return error;
}

async function train(xs, ys, numIterations) {
    for (let iter = 0; iter < numIterations; iter++) {
        optimiser.minimize(() => {
            const pred = predict(xs, degree);
            return loss(pred, ys)
        })
    }
    await tf.nextFrame();
}

async function learnCoefficients() {
    await train(xs, ys, numIterations);
}

parseCSV(file().training).then((e) => {
    normalizeData();
    learnCoefficients().then((pred) => {
        var p = tf.tensor1d([1626464336]);
        const xs = p.sub(i_min).div(i_max.sub(i_min))
        predict(xs, degree).data().then((es) => {
            var pr = tf.tensor1d(es)
            var prediction = pr.mul(l_max.sub(l_min)).add(l_min)
            prediction.data().then((e) => {
                console.log(e)
            })
        })
    })
})