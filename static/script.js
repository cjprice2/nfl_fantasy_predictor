document.addEventListener('DOMContentLoaded', function () {
  loadTeams();
  
  // Load players when a team is selected
  document.getElementById('team-select').addEventListener('change', function () {
      loadPlayers();
      const teamId = document.getElementById('team-select').value;
      if (teamId) {
          loadWeeks(teamId); // Load weeks when a team is selected
      }
  });


  // Calculate and predict fantasy points when the button is clicked
  document.getElementById('predict-btn').addEventListener('click', predictPoints);
});

// Load NFL teams into the dropdown
function loadTeams() {
    fetch('/teams')
        .then(response => response.json())
        .then(data => {
            const teamSelect = document.getElementById('team-select');
            teamSelect.innerHTML = '<option value="">--Select a Team--</option>'; // Default option
            const activeTeams = [
                'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
                'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
            ];
            data.forEach(team => {
                if (activeTeams.includes(team.team_abbr)) {  // Check if the team is active
                    const option = document.createElement('option');
                    option.value = team.team_abbr;
                    option.text = team.team_name;
                    teamSelect.add(option);
                }
            });
        })
        .catch(error => console.error('Error loading teams:', error));
}

// Load players based on the selected team
function loadPlayers() {
  const teamId = document.getElementById('team-select').value;
  fetch(`/players/${teamId}`)
      .then(response => response.json())
      .then(data => {
          const playerSelect = document.getElementById('player-select');
          playerSelect.innerHTML = '<option value="">--Select a Player--</option>'; // Default option
          data.forEach(player => {
              const option = document.createElement('option');
              option.value = player.player_id;
              option.text = player.player_name;
              playerSelect.add(option);
          });
      })
      .catch(error => console.error('Error loading players:', error));
}

// Load weeks and opponents into the dropdown
function loadWeeks(teamId) {
  fetch('/weeks')
      .then(response => response.json())
      .then(weeks => {
          const weekSelect = document.getElementById('week-select');
          weekSelect.innerHTML = ''; // Clear previous options

          const weekPromises = weeks.map(week => {
              return fetch(`/opponents/${teamId}/${week}`)
                  .then(response => {
                      if (!response.ok) {
                          throw new Error(`Week ${week}: Bye or no opponent.`);
                      }
                      return response.json();
                  })
                  .then(opponentData => {
                      const opponent = opponentData.opponent ? opponentData.opponent : 'Unknown';
                      return { week, opponent }; // Return an object with week and opponent
                  })
                  .catch(error => {
                      console.log(error.message); // Log the week if it has no opponent (bye week)
                      return null; // Return null for weeks with no opponents (e.g., bye weeks)
                  });
          });

          // Wait for all the opponent requests to complete
          Promise.all(weekPromises)
              .then(results => {
                  // Filter out any null results (bye weeks)
                  const validWeeks = results.filter(result => result !== null);

                  // Sort the results by week number to ensure correct order
                  validWeeks.sort((a, b) => a.week - b.week);

                  // Add the sorted weeks and opponents to the dropdown
                  validWeeks.forEach(result => {
                      const option = document.createElement('option');
                      option.value = result.week;
                      if (result.opponent === 'Unknown') {
                          option.text = `Week ${result.week} (2024) - Bye`;
                      } else {
                          option.text = `Week ${result.week} (2024) vs ${result.opponent}`;
                      }
                      weekSelect.add(option);
                  });
              })
              .catch(error => console.error('Error loading opponents:', error));
      })
      .catch(error => console.error('Error loading weeks:', error));
}


// Predict fantasy points based on the selected player, team, week, and format
function predictPoints() {
  const playerId = document.getElementById('player-select').value;
  const teamId = document.getElementById('team-select').value;
  const week = document.getElementById('week-select').value;
  const format = document.querySelector('input[name="format"]:checked').value;

  if (!playerId || !teamId || !week) {
      alert('Please select a team, player, and week.');
      return;
  }

  fetch('/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({
          player_id: playerId,
          team_id: teamId,
          week: week,  // Pass the selected week
          format: format
      })
  })
      .then(response => response.json())
      .then(data => {
          const resultDiv = document.getElementById('result');
          resultDiv.textContent = `Predicted Fantasy Points: ${data.predicted_fantasy_points.toFixed(2)}`;
      })
      .catch(error => console.error('Error predicting points:', error));
}
