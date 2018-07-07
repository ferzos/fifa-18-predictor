const tf = require('@tensorflow/tfjs');
const axios = require('axios');
const fs = require('fs');

// Load the binding:
require('@tensorflow/tfjs-node');

// Set the backend to TensorFlow:
tf.setBackend('tensorflow');

const jsonData = []
const home_team = 'England'
const away_team = 'Panama'

async function make(cb) {
  let data = await axios.get('http://worldcup.sfg.io/matches')
  data = data.data
  
  data.filter( match => match.status === 'completed').map( match => {
    const home_team_stats = match.home_team_statistics
    const away_team_stats = match.away_team_statistics
    const dataObject = {}

    Object.entries(home_team_stats).filter( entries => entries[0] !== 'country' && entries[0] !== 'starting_eleven' && entries[0] !== 'substitutes' && entries[0] !== 'tactics').map( entries => dataObject[`home_${entries[0]}`] = entries[1])
    Object.entries(away_team_stats).filter( entries => entries[0] !== 'country' && entries[0] !== 'starting_eleven' && entries[0] !== 'substitutes' && entries[0] !== 'tactics').map( entries => dataObject[`away_${entries[0]}`] = entries[1])
    dataObject['win'] = match.home_team.country === match.winner ? 'home' : 'away'
    
    jsonData.push(dataObject)
  })
  
  console.log(`Total data: ${jsonData.length}`);
  

  fs.writeFile('data.json', JSON.stringify(jsonData), (err) => {  
    // throws an error, you could also catch it here
    if (err) throw err;
  
   // success case, the file was saved
    console.log(`Done writing`);
  });

  cb(home_team, away_team)
}

async function run(home, away) {
  const trainingData = tf.tensor2d(jsonData.map( match => [
    match.home_attempts_on_goal,
    match.home_on_target,
    match.home_off_target,
    match.home_blocked,
    match.home_woodwork,
    match.home_corners,
    match.home_offsides,
    match.home_ball_possession,
    match.home_pass_accuracy,
    match.home_num_passes,
    match.home_passes_completed,
    match.home_distance_covered,
    match.home_balls_recovered,
    match.home_tackles,
    match.home_clearances,
    match.home_yellow_cards,
    match.home_red_cards,
    match.home_fouls_committed,
    match.away_attempts_on_goal,
    match.away_on_target,
    match.away_off_target,
    match.away_blocked,
    match.away_woodwork,
    match.away_corners,
    match.away_offsides,
    match.away_ball_possession,
    match.away_pass_accuracy,
    match.away_num_passes,
    match.away_passes_completed,
    match.away_distance_covered,
    match.away_balls_recovered,
    match.away_tackles,
    match.away_clearances,
    match.away_yellow_cards,
    match.away_red_cards,
    match.away_fouls_committed,
  ]));

  const outputData = tf.tensor2d(jsonData.map( match => [
    match.win === 'home' ? 1 : 0,
    match.win === 'away' ? 1 : 0 
  ]))

  const model = tf.sequential()

  model.add(tf.layers.dense({
    inputShape: [36],
    activation: 'sigmoid',
    units: 5,
  }))

  model.add(tf.layers.dense({
    inputShape: [5],
    activation: 'sigmoid',
    units: 2,
  }))

  model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 2,
  }))

  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.06)
  })

  const reqData = await axios.get('http://worldcup.sfg.io/matches')
  const arrayData = reqData.data.filter( match => match.status === 'completed').reverse()
  
  const last_home_match = arrayData.find(match => match.home_team_country === home || match.away_team_country === home)
  const last_away_match = arrayData.find(match => match.home_team_country === away || match.away_team_country === away)
  const last_home_match_stats = last_home_match.home_team_country === home ? last_home_match.home_team_statistics : last_home_match.away_team_statistics
  const last_away_match_stats = last_away_match.home_team_country === away ? last_away_match.home_team_statistics : last_away_match.away_team_statistics
  delete last_home_match_stats.starting_eleven
  delete last_home_match_stats.substitutes
  delete last_away_match_stats.starting_eleven
  delete last_away_match_stats.substitutes
  
  const inputData = tf.tensor2d([[
    last_home_match_stats.attempts_on_goal,
    last_home_match_stats.on_target,
    last_home_match_stats.off_target,
    last_home_match_stats.blocked,
    last_home_match_stats.woodwork,
    last_home_match_stats.corners,
    last_home_match_stats.offsides,
    last_home_match_stats.ball_possession,
    last_home_match_stats.pass_accuracy,
    last_home_match_stats.num_passes,
    last_home_match_stats.passes_completed,
    last_home_match_stats.distance_covered,
    last_home_match_stats.balls_recovered,
    last_home_match_stats.tackles,
    last_home_match_stats.clearances,
    last_home_match_stats.yellow_cards,
    last_home_match_stats.red_cards,
    last_home_match_stats.fouls_committed,
    last_away_match_stats.attempts_on_goal,
    last_away_match_stats.on_target,
    last_away_match_stats.off_target,
    last_away_match_stats.blocked,
    last_away_match_stats.woodwork,
    last_away_match_stats.corners,
    last_away_match_stats.offsides,
    last_away_match_stats.ball_possession,
    last_away_match_stats.pass_accuracy,
    last_away_match_stats.num_passes,
    last_away_match_stats.passes_completed,
    last_away_match_stats.distance_covered,
    last_away_match_stats.balls_recovered,
    last_away_match_stats.tackles,
    last_away_match_stats.clearances,
    last_away_match_stats.yellow_cards,
    last_away_match_stats.red_cards,
    last_away_match_stats.fouls_committed
  ]])

  const startTime = Date.now()
  model.fit(trainingData, outputData, {epochs: 100})
        .then(history => {
          console.log(`Done fitting in: ${Date.now() - startTime}s`)
          model.predict(inputData).print()
          console.log(`${home}, ${away}`);
        })
}

make(run)
