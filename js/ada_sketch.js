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
    var canvas = createCanvas(800,550);

    canvas.parent('ada_sketch-holder');

    pbrSlider = createSlider(1,60,10);
    pbrSlider.position(170,10);
    pbrSlider.remove();

    // cwLogData = table.getRows();
    print(table.getRowCount() + ' total rows in table');
    print(table.getColumnCount() + ' total columns in table');

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

    background(colorBackground);

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

    var row1 = 15;
    var row2 = 350;
    // var sourceWaterObj = new AnimatedBar("source-water","Source Water", 1,colorBlue).waterDisplay();
    new AnimatedBar("source-water","Source Water",    row1,0,colorBlue).waterDisplay();
    new AnimatedBar("desert-water","Desert Water",    row1,1,colorBlue).waterDisplay();
    new AnimatedBar("plains-water","Plains Water",    row1,2,colorBlue).waterDisplay();
    new AnimatedBar("jungle-water","Jungle Water",    row1,3,colorBlue).waterDisplay();
    new AnimatedBar("wetlands-water","Wetlands Water",row1, 4,colorBlue).waterDisplay();

    new AnimatedBar("plains-plants-level4-alive", "PlainL4", row1, 6, colorGreen).display();
    new AnimatedBar("plains-plants-level3-alive", "PlainL3", row1, 7, colorGreen).display();
    new AnimatedBar("plains-plants-level2-alive", "PlainL2", row1, 8, colorGreen).display();
    new AnimatedBar("plains-plants-level1-alive", "PlainL1", row1, 9, colorGreen).display();

    new AnimatedBar("plains-creatures-level4-alive", "PlainL4", row1, 10, colorRed).display();
    new AnimatedBar("plains-creatures-level3-alive", "PlainL3", row1, 11, colorRed).display();
    new AnimatedBar("plains-creatures-level2-alive", "PlainL2", row1, 12, colorRed).display();
    new AnimatedBar("plains-creatures-level1-alive", "PlainL1", row1, 13, colorRed).display();

    new AnimatedBar("desert-plants-level4-alive", "DesertL4", row1, 15, colorGreen).display();
    new AnimatedBar("desert-plants-level3-alive", "DesertL3", row1, 16, colorGreen).display();
    new AnimatedBar("desert-plants-level2-alive", "DesertL2", row1, 17, colorGreen).display();
    new AnimatedBar("desert-plants-level1-alive", "DesertL1", row1, 18, colorGreen).display();

    new AnimatedBar("desert-creatures-level4-alive", "DesertL4", row1, 19, colorRed).display();
    new AnimatedBar("desert-creatures-level3-alive", "DesertL3", row1, 20, colorRed).display();
    new AnimatedBar("desert-creatures-level2-alive", "DesertL2", row1, 21, colorRed).display();
    new AnimatedBar("desert-creatures-level1-alive", "DesertL1", row1, 22, colorRed).display();


    new AnimatedBar("jungle-plants-level4-alive", "JungleL4", row2, 6, colorGreen).display();
    new AnimatedBar("jungle-plants-level3-alive", "JungleL3", row2, 7, colorGreen).display();
    new AnimatedBar("jungle-plants-level2-alive", "JungleL2", row2, 8, colorGreen).display();
    new AnimatedBar("jungle-plants-level1-alive", "JungleL1", row2, 9, colorGreen).display();
    new AnimatedBar("jungle-creatures-level4-alive", "JungleL4", row2, 10, colorRed).display();
    new AnimatedBar("jungle-creatures-level3-alive", "JungleL3", row2, 11, colorRed).display();
    new AnimatedBar("jungle-creatures-level2-alive", "JungleL2", row2, 12, colorRed).display();
    new AnimatedBar("jungle-creatures-level1-alive", "JungleL1", row2, 13, colorRed).display();


    new AnimatedBar("wetlands-plants-level4-alive", "WetlandsL4", row2, 15, colorGreen).display();
    new AnimatedBar("wetlands-plants-level3-alive", "WetlandsL3", row2, 16, colorGreen).display();
    new AnimatedBar("wetlands-plants-level2-alive", "WetlandsL2", row2, 17, colorGreen).display();
    new AnimatedBar("wetlands-plants-level1-alive", "WetlandsL1", row2, 18, colorGreen).display();

    new AnimatedBar("wetlands-creatures-level4-alive", "WetlandsL4", row2, 19, colorRed).display();
    new AnimatedBar("wetlands-creatures-level3-alive", "WetlandsL3", row2, 20, colorRed).display();
    new AnimatedBar("wetlands-creatures-level2-alive", "WetlandsL2", row2, 21, colorRed).display();
    new AnimatedBar("wetlands-creatures-level1-alive", "WetlandsL1", row2, 22, colorRed).display();


}

function AnimatedBar(columnName, displayName, horizontalPosition, verticalPosition,fillColor){
    this.columnName = String(columnName);
    this.displayName = String(displayName);
    this.verticalPosition = Number(verticalPosition);
    this.fillColor = fillColor;
    this.xPos = Number(horizontalPosition);
    this.yPos = 60+this.verticalPosition * 20;
    this.barHeight = 20;
    this.barXShift = 140;
    this.tempSource = table.getColumn(this.columnName);

    this.waterDisplay = function (){
        fill(this.fillColor);
        rect(this.xPos+this.barXShift,this.yPos-this.barHeight/2-5,this.tempSource[i]*200,this.barHeight, 0,5,5,0);
        textAlign(RIGHT);
        text(this.displayName, this.xPos+this.barXShift-40,this.yPos);

        fill(color(88,110,117)); //for text
        text(nf(this.tempSource[i],1,2)+" ",this.xPos+this.barXShift,this.yPos);

    }
    this.display = function (){
        fill(this.fillColor);
        rect(this.xPos+this.barXShift,this.yPos-this.barHeight/2-5,this.tempSource[i]*5,this.barHeight,0,5,5, 0);
        textAlign(RIGHT);
        text(this.displayName, this.xPos+this.barXShift-40,this.yPos);

        fill(color(88,110,117)); //for text
        text(this.tempSource[i]+" ",this.xPos+this.barXShift,this.yPos);

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
