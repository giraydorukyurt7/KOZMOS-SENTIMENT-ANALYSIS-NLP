const csv = require('csv-parser')
const fs = require('fs')
const results = [];

fs.createReadStream('../Generated_files/Log_model_scores_df.csv')
  .pipe(csv())
  .on('data', (data) => results.push(data))
  .on('end', () => {
    console.log(results);
    for(let i = 0; i < results.length; i++){
        let Metric = results[i].Metric;
        let precision = results[i].precision;
        let recall = results[i].recall;
        let f1_score = results[i].f1_score;
        let support = results[i].support;
    }
  });