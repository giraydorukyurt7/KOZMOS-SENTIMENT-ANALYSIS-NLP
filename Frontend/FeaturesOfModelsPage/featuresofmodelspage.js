const csv = require('csv-parser'); // Include the csv-parser module
const fs = require('fs'); // Include the file system module
const path = require('path'); // Include the path module

// Use __dirname to get the absolute path of the current directory
const filePath = path.join(__dirname, '../../Generated_files/Log_model_scores_df.csv');

console.log('Trying to read from:', filePath); // Log the file path for debugging

const results = []; // Array to store the parsed data

fs.createReadStream(filePath)
  .pipe(csv())
  .on('data', (data) => results.push(data)) // Add each row of data to the results array
  .on('end', () => {
    console.log(results); // Log the entire dataset to the console
    for(let i=0;i<results.length;i++){
      let metric = results[i].metrics
    }
  })
  .on('error', (err) => {
    console.error('Error reading the file:', err.message);
  });

  console.log(metric);