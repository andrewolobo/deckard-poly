import fs from "fs";

const training = fs.readFileSync( "./xlsx/data.csv", "utf8");
const test = fs.readFileSync( "./xlsx/test.data.csv", "utf8");

function file(){
    return {training, test}
}

export default file;