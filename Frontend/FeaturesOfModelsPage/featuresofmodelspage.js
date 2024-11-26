// READ CSV WITH FETCH
function loadCSV(filePath, title) {
    fetch(filePath)
      .then(response => response.text())
      .then(data => {
        const lines = data.split('\n'); // Split csv lines
        let tableHTML = `<h3>${title}</h3><table border="1"><tr><th>#</th><th>Metric</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>`; // titles
  
        // Create Table
        for (let i = 1; i < lines.length; i++) {
          const line = lines[i].trim(); // Clean titles and spaces
          if (line) {  // If line is not empty
            const columns = line.split(','); // Split the line according to ','
            
            //Change empty cells with 'NaN'
            for (let j=1;j<columns.length;j++){
              if(columns[j].trim()===''){
                columns[j] ='NaN';
              }
            }
            
            // Add new table line for every line
            tableHTML += `<tr>
                            <td>${columns[0]}</td>
                            <td>${columns[1]}</td>
                            <td>${columns[2]}</td>
                            <td>${columns[3]}</td>
                            <td>${columns[4]}</td>
                            <td>${columns[5]}</td>
                          </tr>`;
          }
        }
  
        tableHTML += '</table>'; // close table
  
        // print table
        document.getElementById('output').innerHTML += tableHTML; // to not delete first table when adding second one.
      })
      .catch(error => console.error('Error reading the CSV file:', error));
  }
  
  // 1st csv file
  loadCSV('../../Generated_files/Log_model_scores_df.csv', 'Logistic Regression Model Scores');
  
  // 2nd csv file
  loadCSV('../../Generated_files/rf_model_scores_df.csv', 'Random Forests Model Scores');
  