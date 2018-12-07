var BIOMES = ['Desert_Water', 'Plains_Water', 'Jungle_Water', 'Wetlands_Water']
var PLANTS = ['Desert_Plants', 'Plains_Plants', 'Jungle_Plants', 'Wetlands_Plants']
var LEVELS = ['lv1', 'lv2', 'lv3', 'lv4']
var COLORS = ['blue', 'red', 'purple', 'cyan'];


/*
Sort out the video stuffs here
 */
var vid = document.getElementById("globalView");
var speed = 3.0;
vid.playbackRate = speed;


/*
D3v4 magic goes in here
*/



var margin = {top: 30, right: 20, bottom: 30, left: 50};

//====================================================================
// USEFUL FUNCTIONS

// Get the data
var waterLevelsAndPeriods = null;
var fix_seconds = [];
var seen = [];
var zoomedChart = null;
var mainChart = null;
var plantsChart = null;
var plotted_main_periods = false;

function pad(value) {
    if(value < 10) {
        return '0' + value;
    } else {
        return value;
    }
}

function get_time_string(seconds) {
    var minutes = Math.floor(seconds / 60);
    var sec = seconds - minutes * 60
    return pad(minutes) + ':' + pad(sec);
}

d3.csv("data/input_file.csv", function(error, data) {

    var mapped_data = data.map(function(d) {
        var dte = new Date(2014, 4, 1);
        var u = +dte;
        var seconds = parseInt(d.seconds);
        var newU = u + seconds*1000;
        var newD = new Date(newU);

        var datapoint = {};
        datapoint.seconds = d.seconds;
        datapoint.start = d.start;
        datapoint.stop = d.stop;
        datapoint.date = newD;
        datapoint.Desert_Water = +parseFloat(d.Desert_Water);
        datapoint.Jungle_Water = +parseFloat(d.Jungle_Water);
        datapoint.Wetlands_Water = +parseFloat(d.Wetlands_Water);
        datapoint.Plains_Water = +parseFloat(d.Plains_Water);
        datapoint.Desert_Plants = +parseFloat(d.Desert_Plants);
        datapoint.Jungle_Plants = +parseFloat(d.Jungle_Plants);
        datapoint.Wetlands_Plants = +parseFloat(d.Wetlands_Plants);
        datapoint.Plains_Plants = +parseFloat(d.Plains_Plants);
        datapoint.vid_sec = +parseInt(d.vid_sec);
        datapoint.important = +parseInt(d.interesting);

        PLANTS.forEach(function(plant) {
            LEVELS.forEach(function(level) {
                var value = plant.replace('Plants', level);
                datapoint[value] = +parseInt(d[value]);
            });
        });


        var top = ['top1', 'top2', 'top3', 'top4'];
        var bottom = ['bottom1', 'bottom2', 'bottom3', 'bottom4'];

        datapoint.top = top.map( function(x) { return (x=='None' ? null : parseInt(d[x]) + 1) }).filter( (v) => v ).map( function(x) { return x-1; });
        datapoint.bottom = bottom.map( function(x) { return (x=='None' ? null : parseInt(d[x]) + 1) }).filter( (v) => v ).map( function(x) { return x-1; });

        if ((seen.indexOf(d.vid_sec) == -1)) {
            seen.push(d.vid_sec)
            fix_seconds.push(seconds)
        }

        return datapoint;
    });

    waterLevelsAndPeriods = mapped_data;

    zoomedChart = new LineChart(
        data = waterLevelsAndPeriods,
        width = $('#zoomed_chart').width() - margin.left - margin.right,
        margin = margin,
        height = $('#zoomed_chart').height() - margin.top - margin.bottom,
        colors = COLORS,
        element = '#zoomed_chart'
    );
    zoomedChart.plotLegend(BIOMES);

    mainChart = new LineChart(
        data = waterLevelsAndPeriods,
        width = $('#full_chart').width() - margin.left - margin.right,
        margin = margin,
        height = $('#full_chart').height() - margin.top - margin.bottom,
        colors = COLORS,
        element = '#full_chart'
    );
    mainChart.plotBiomes(BIOMES);
    mainChart.plotLegend(BIOMES);

    plantsChart = new LineChart(
        data = waterLevelsAndPeriods,
        width = $('#plants_chart').width() - margin.left - margin.right,
        margin = margin,
        height = $('#plants_chart').height() - margin.top - margin.bottom,
        colors = COLORS,
        element = '#plants_chart'
    );
    plantsChart.plotLegend( getActivePlants() );

    // attach the click handlers:
    var array_flipped={};
    $.each(fix_seconds, function(i, el) {
        array_flipped[el]=parseInt(i);
    });
    var data = [array_flipped];
    mainChart.attachTimeClickHandler(updateVidTime, data);
    zoomedChart.attachTimeClickHandler(updateVidTime, data);
    plantsChart.attachTimeClickHandler(updateVidTime, data);
});

function updateVidTime(time, data) {
    var vid = document.getElementById("globalView");
    var array_flipped = data[0];
    var sec = parseInt(time);
    while (!(sec in array_flipped)) {
        sec = sec + 1;
    }
    pause_vid();
    vid.currentTime = array_flipped[sec];
}

function updateCharts(time) {

    if ((time <= 1) | (plotted_main_periods == false)) {
        waterLevelsAndPeriods.forEach(function(x) {
            var time = x.seconds;
            mainChart.plotActiveRegions(time, false);
        });
        plotted_main_periods = true;
    }

    zoomedChart.plotBiomes(BIOMES, time-30, time+70);
    plantsChart.plotBiomes( getActivePlants() , time-30, time+70);

    mainChart.plotTimeTracker(time);
    zoomedChart.plotTimeTracker(time);
    plantsChart.plotTimeTracker(time);

    zoomedChart.plotActiveRegions(time, true);
    plantsChart.plotActiveRegions(time, true);

    mainChart.plotFocus(time);

    plantsChart.plotLegend( getActivePlants() );

    $('#top_biomes').empty();
    waterLevelsAndPeriods[time].top.forEach( function(x) {
        $('#top_biomes').append($('<li>'+BIOMES[x].replace('_Water', ' ')+'</li>').css('color', COLORS[x]));
    });
    $('#bottom_biomes').empty();
    waterLevelsAndPeriods[time].bottom.forEach( function(x) {
        $('#bottom_biomes').append($('<li>'+BIOMES[x].replace('_Water', ' ')+'</li>').css('color', COLORS[x]));
    });
}

vid.ontimeupdate = function() {
    var time = fix_seconds[parseInt(vid.currentTime)]+1;
    updateCharts(time);
    $('#time').text('Video Time: ' + get_time_string(time))
};

/* Plotting controls*/

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function getActivePlants() {
    var plant_checkboxes = $('.plant_checkbox');
    return (plant_checkboxes.map(function(i, box) {
        var checkbox_id = box.id;
        if (box.checked) {
            checkbox_id = checkbox_id.replace('checkbox_', '');
            checkbox_id = checkbox_id.replace('_plant_', '_lv');
            return capitalizeFirstLetter(checkbox_id);
        }
    })).toArray();
};

$('.plant_checkbox').each(function(i, box) {
    $(box).change(function() {
        var time = fix_seconds[parseInt(vid.currentTime)];
        plantsChart.plotBiomes( getActivePlants() , time-30, time+70);
        plantsChart.plotLegend( getActivePlants() );
    });
});

$('.plant_checkbox_all').each(function(i, box) {
    $(box).change(function() {
        var id = box.id;
        var biome = id.replace('checkbox_all_', '');
        var other_boxes = [1,2,3,4].map(function(x) { return '#checkbox_' + biome + '_plant_' + x;});

        other_boxes.forEach( function (value) {
            $(value)[0].checked = box.checked;
        });
        var time = fix_seconds[parseInt(vid.currentTime)];
        plantsChart.plotBiomes( getActivePlants() , time-30, time+70);
        plantsChart.plotLegend( getActivePlants() );
    });
});

/* video controls */
$(document).keydown(function(e) {

    switch(e.which) {
        case 37: // left
            var vid = $('#globalView')[0];
            var time = vid.currentTime
            vid.currentTime = time - 2;
            // vid.currentTime = (start - 2);
            break;

        case 38: // up
            break;

        case 39: // right
            var vid = $('#globalView')[0];
            var time = vid.currentTime
            vid.currentTime = time + 2;
            // vid.currentTime = (start + 2);
            break;

        case 40: // down
            break;

        default: return; // exit this handler for other keys
    }
    e.preventDefault(); // prevent the default action (scroll / move caret)
});

$('#globalView').click(function() {
    this.paused ? play_vid() : pause_vid();
});

function pause_vid() {
    var global_view = $('#globalView')[0];
    $('#play-pause')[0].setAttribute('playing', 'false');
    $('#play-pause').html("<span class='glyphicon glyphicon-play'></span> Play");
    global_view.pause();
}

function play_vid() {
    var global_view = $('#globalView')[0];
    $('#play-pause')[0].setAttribute('playing', 'true');
    $('#play-pause').html("<span class='glyphicon glyphicon-pause'></span> Pause");
    global_view.play();
}

$('#play-pause').click(function() {
    var playing = this.getAttribute('playing');
    if (playing == 'true') {
        pause_vid();
    }
    else {
        play_vid();
    }
});
