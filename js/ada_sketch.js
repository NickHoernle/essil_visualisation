var table; // data loaded from csv
var cwLogData;
var sourceWater = [];
var i = 0;
var isPlaying = false;

function preload(){
    table = loadTable('data/20181208-trial1-converted-v5.csv','csv','header');
}

function setup() {
    // put setup code here
    var canvas = createCanvas(800,600);

    canvas.parent('ada_sketch-holder');

    pbrSlider = createSlider(1,60,10);
    pbrSlider.position(170,10);
    pbrSlider.remove();

    // cwLogData = table.getRows();
    // print(table.getRowCount() + ' total rows in table');
    // print(table.getColumnCount() + ' total columns in table');

    sourceWater = table.getColumn('source-water');
    sysTime = table.getColumn("system-time");

    // create button for playing
    // playButton = createButton('Play');
    // playButton.position(40, 10);
    // playButton.mousePressed(toggleIsPlaying);

    // create button for decrement timestep
    // decrementButton = createButton("&lt;");
    // decrementButton.position(10,10);
    // decrementButton.mousePressed(decrementTime);
    //
    // // create button for increment timestep
    // incrementButton = createButton("&gt;");
    // incrementButton.position(90,10);
    // incrementButton.mousePressed(incrementTime);

}

function draw() {
    // put drawing code here

    // colors
    var colorOrange = color(203,75,22);
    var colorBackground = color(244, 244, 244);
    var colorWhite = color(255);
    var colorBlue = color(38,139,210);
    var colorGreen = color(133,153,0);
    var colorRed = color(220,50,47);

    // background(colorBackground);
    background(255);

    // playback rate
    fill(0);
    textAlign(LEFT);
    // text("playback rate: "+str(pbrSlider.value())+"x", 320,22);
    frameRate(pbrSlider.value()); // set the framerate

    // textstring of current time
    textAlign(LEFT);
    fill(colorOrange);
    // text(str(int(i/60))+"m"+str(int(i)%60+"s"), 120, 23);


    if (i >= sourceWater.length-1){ // check if out of data
        i=0; //reset time to 0
        isPlaying = false; //pause playing
    }

    if (isPlaying) {i++;} // increment time

    var column_1 = 15;
    var column_2 = 350;
    fill(colorBlue);
    textSize(18);
    text("Water", column_1,30);
    // var sourceWaterObj = new AnimatedBar("source-water","Source Water", 1,colorBlue).waterDisplay();
    new AnimatedBar("source-water","Source", column_1,0,colorBlue).waterDisplay();
    new AnimatedBar("desert-water","Desert", column_1,1,colorBlue).waterDisplay();
    new AnimatedBar("plains-water","Grasslands", column_1,2,colorBlue).waterDisplay();
    new AnimatedBar("jungle-water","Jungle", column_1,3,colorBlue).waterDisplay();
    new AnimatedBar("wetlands-water","Wetlands", column_1, 4,colorBlue).waterDisplay();

    var checkBoxBarGraphs = ["show_desert_bar","show_plains_bar","show_jungle_bar","show_wetlands_bar"];
    // var allBiomes = [desertEnv,grasslandsEnv, jungleEnv, wetlandsEnv];
    // var checkedBiomes = [];
    var numberChecked = 0;
    for ( let k = 0; k < checkBoxBarGraphs.length; k++){
        if(checkIfChecked(checkBoxBarGraphs[k]) == 1){
            numberChecked ++;
            // checkedBiomes += allBiomes[i];
        }
    }

    function checkIfChecked(string){
        if(document.getElementById(string).checked){
            return 1;
        }
        else{
            return 0;
        }
    }
    if(numberChecked == 1){
        if(checkIfChecked("show_jungle_bar")){
            new PlotBars(jungleEnv,posA1).display();
        } else if (checkIfChecked("show_desert_bar")){
            new PlotBars(desertEnv,posA1).display();
        } else if (checkIfChecked("show_plains_bar")){
            new PlotBars(grasslandsEnv,posA1).display();
        } else if (checkIfChecked("show_wetlands_bar")){
            new PlotBars(wetlandsEnv,posA1).display();
        }
    } else if (numberChecked == 2) {
        if(checkIfChecked("show_jungle_bar") && checkIfChecked("show_desert_bar")){
            new PlotBars(desertEnv,posB1).display();
            new PlotBars(jungleEnv,posB2).display();
        } else if (checkIfChecked("show_desert_bar") && checkIfChecked("show_plains_bar")){
            new PlotBars(desertEnv,posB1).display();
            new PlotBars(grasslandsEnv,posB2).display();
        } else if (checkIfChecked("show_desert_bar") && checkIfChecked("show_wetlands_bar")){
            new PlotBars(desertEnv,posB1).display();
            new PlotBars(wetlandsEnv,posB2).display();
        } else if (checkIfChecked("show_jungle_bar") && checkIfChecked("show_plains_bar")){
            new PlotBars(jungleEnv,posB2).display();
            new PlotBars(grasslandsEnv,posB1).display();
        } else if (checkIfChecked("show_jungle_bar") && checkIfChecked("show_wetlands_bar")){
            new PlotBars(jungleEnv,posB1).display();
            new PlotBars(wetlandsEnv,posB2).display();
        } else if (checkIfChecked("show_wetlands_bar") && checkIfChecked("show_plains_bar")){
            new PlotBars(wetlandsEnv,posB2).display();
            new PlotBars(grasslandsEnv,posB1).display();
        }
    } else if (numberChecked == 3){
        if(checkIfChecked("show_jungle_bar")== 0){
            new PlotBars(grasslandsEnv,posC1).display();
            new PlotBars(desertEnv,posC3).display();
            new PlotBars(wetlandsEnv,posC2).display();
        } else if (checkIfChecked("show_desert_bar")== 0){
            new PlotBars(grasslandsEnv,posC1).display();
            new PlotBars(jungleEnv,posC2).display();
            new PlotBars(wetlandsEnv,posC4).display();
        } else if (checkIfChecked("show_plains_bar")== 0){
            new PlotBars(desertEnv,posC1).display();
            new PlotBars(jungleEnv,posC2).display();
            new PlotBars(wetlandsEnv,posC3).display();
        } else if (checkIfChecked("show_wetlands_bar")== 0){
            new PlotBars(grasslandsEnv,posC1).display();
            new PlotBars(desertEnv,posC3).display();
            new PlotBars(jungleEnv,posC2).display();
        }

    }else if (numberChecked == 4){
        new PlotBars(grasslandsEnv,posC1).display();
        new PlotBars(desertEnv,posC3).display();
        new PlotBars(jungleEnv,posC2).display();
        new PlotBars(wetlandsEnv,posC4).display();
    } else {
        new PlotBars(grasslandsEnv,posC1).display();
        new PlotBars(desertEnv,posC3).display();
        new PlotBars(jungleEnv,posC2).display();
        new PlotBars(wetlandsEnv,posC4).display();
    }
    // print(i);

}
function PlotBars(biomeNameEnv,positionVal){
    this.biome = biomeNameEnv;
    this.pos = positionVal;
    this.plantsPosY = this.pos.plotY+40;
    this.creaturesPosY = this.pos.plotY+this.pos.height/2+20;

    this.display = function(){
        fill(255);
        stroke(this.biome.color);
        strokeWeight(2);
        rect(this.pos.plotX,this.pos.plotY,this.pos.width,this.pos.height);
        strokeWeight(0);
        textAlign(LEFT);
        fill(color(this.biome.color));
        textSize(18);
        text(String(this.biome.textName),this.pos.plotX+10,this.pos.plotY+20);

        // text labels
        textSize(14);
        fill(color(133,153,0));
        text("Plants",this.pos.plotX+10,this.plantsPosY);
        fill(color(220,50,47));
        text("Animals",this.pos.plotX+10,this.creaturesPosY);

        textSize(12);

        new AnimatedBar(this.biome.dataName+"-plants-level4-alive", "Level 4", this.pos.plotX, this.plantsPosY, color(133,153,0),this.pos.barHeight,this.pos.widthMultiplier).display();
        new AnimatedBar(this.biome.dataName+"-plants-level3-alive", "Level 3", this.pos.plotX, this.plantsPosY+this.pos.barHeight*1, color(133,153,0),this.pos.barHeight,this.pos.widthMultiplier).display();
        new AnimatedBar(this.biome.dataName+"-plants-level2-alive", "Level 2", this.pos.plotX, this.plantsPosY+this.pos.barHeight*2, color(133,153,0),this.pos.barHeight,this.pos.widthMultiplier).display();
        new AnimatedBar(this.biome.dataName+"-plants-level1-alive", "Level 1", this.pos.plotX, this.plantsPosY+this.pos.barHeight*3, color(133,153,0),this.pos.barHeight,this.pos.widthMultiplier).display();

        new AnimatedBar(this.biome.dataName+"-creatures-level4-alive", "Level 4", this.pos.plotX, this.creaturesPosY, color(220,50,47),this.pos.barHeight,this.pos.widthMultiplier).display();
        new AnimatedBar(this.biome.dataName+"-creatures-level3-alive", "Level 3", this.pos.plotX, this.creaturesPosY+this.pos.barHeight*1, color(220,50,47),this.pos.barHeight,this.pos.widthMultiplier).display();
        new AnimatedBar(this.biome.dataName+"-creatures-level2-alive", "Level 2", this.pos.plotX, this.creaturesPosY+this.pos.barHeight*2, color(220,50,47),this.pos.barHeight,this.pos.widthMultiplier).display();
        new AnimatedBar(this.biome.dataName+"-creatures-level1-alive", "Level 1", this.pos.plotX, this.creaturesPosY+this.pos.barHeight*3, color(220,50,47),this.pos.barHeight,this.pos.widthMultiplier).display();

    }
}

// object calls from PlotBars
var grasslandsEnv = {
    textName: "Grasslands",
    dataName: "plains",
    color: "rgb(42,161,152)"
};
var desertEnv = {
    textName: "Desert",
    dataName: "desert",
    color: "rgb(181,137,0)"
};
var jungleEnv = {
    textName: "Jungle",
    dataName: "jungle",
    color: "rgb(211,54,130)"
};
var wetlandsEnv = {
    textName: "Wetlands",
    dataName: "wetlands",
    color: "rgb(108,113,196)"
};

// dimensions for positions of graphs
var posA1 = {
    plotX: 15,
    plotY: 150,
    width: 770,
    height: 440,
    barHeight: 40,
    widthMultiplier: 2
};
var posB1 = {
    plotX: 15,
    plotY: 150,
    width: 375,
    height: 440,
    barHeight: 40,
    widthMultiplier: 1
};
var posB2 = {
    plotX: 405,
    plotY: 150,
    width: 375,
    height: 440,
    barHeight: 40,
    widthMultiplier: 1
};
var posC1 = {
    plotX: 15,
    plotY: 150,
    width: 375,
    height: 215,
    barHeight: 20,
    widthMultiplier: 1
};
var posC2 = {
    plotX: 405,
    plotY: 150,
    width: 375,
    height: 215,
    barHeight: 20,
    widthMultiplier: 1
};
var posC3 = {
    plotX: 15,
    plotY: 375,
    width: 375,
    height: 215,
    barHeight: 20,
    widthMultiplier: 1
};
var posC4 = {
    plotX: 405,
    plotY: 375,
    width: 375,
    height: 215,
    barHeight: 20,
    widthMultiplier: 1
};

function AnimatedBar(columnName, displayName, horizontalPosition, verticalPosition,fillColor,barHeight, barWidthMultiplier){
    this.columnName = String(columnName);
    this.displayName = String(displayName);
    this.verticalPosition = Number(verticalPosition);
    this.fillColor = fillColor;
    this.xPos = Number(horizontalPosition);
    this.yPos = 60+this.verticalPosition * 20;
    barHeight ? this.barHeight = barHeight : this.barHeight = 20;
    this.barXShift = 140;
    barWidthMultiplier ? this.barWidthMultiplier : this.barWidthMultiplier = 1;
    this.tempSource = table.getColumn(this.columnName);
    this.yPos2 = this.verticalPosition + 15;

    this.waterDisplay = function (){
        fill(this.fillColor);
        stroke(200);
        strokeWeight(1);
        rect(this.xPos+this.barXShift,this.yPos-this.barHeight/2-5,this.tempSource[i]*100,this.barHeight, 0,5,5,0);
        strokeWeight(0);
        textAlign(RIGHT);
        textSize(12);
        text(this.displayName, this.xPos+this.barXShift-40,this.yPos);


        fill(color(88,110,117)); //for numerical value
        text(nf(this.tempSource[i],1,2)+" ",this.xPos+this.barXShift,this.yPos);

    }
    this.display = function (){
        fill(this.fillColor);
        stroke(200);
        strokeWeight(1);
        rect(this.xPos+this.barXShift,this.yPos2-this.barHeight*.60,this.tempSource[i]*3*barWidthMultiplier,this.barHeight,0,5,5,0);
        strokeWeight(0);
        textAlign(RIGHT);
        textSize(12);
        text(this.displayName, this.xPos+this.barXShift-25,this.yPos2);

        fill(color(88,110,117)); //for numerical value
        text(this.tempSource[i]+" ",this.xPos+this.barXShift,this.yPos2);
    }

}

function toggleIsPlaying(){
    if(isPlaying) {
        isPlaying = false;
        playButton.html('Play');
    }else {
        isPlaying = true;
        playButton.html('Pause');
    }
}

// for increment and decrement buttons
function incrementTime(){
    i++;
}
function decrementTime(){
    i--;
}

function set_time(time) {
    i = time;
}
