/**
 * Created by nickhoernle on 2018/10/09.
 */

var format = d3.timeFormat("%M:%S")
var regime_map = {
    "Desert_important_down":   "Desert_regime",
    "Plains_important_down":   "Plains_regime",
    "Jungle_important_down":   "Jungle_regime",
    "Wetlands_important_down": "Wetlands_regime",
    "Desert_important_up":   "Desert_regime",
    "Plains_important_up":   "Plains_regime",
    "Jungle_important_up":   "Jungle_regime",
    "Wetlands_important_up": "Wetlands_regime",
    "Wetlands_Raining": "Wetlands_regime",
    "Desert_Raining": "Desert_regime",
    "Plains_Raining": "Plains_regime",
    "Jungle_Raining": "Jungle_regime"
}

function plot_line(biome, x, y) {
    return d3.line()
        .x(function (d) {
            return x(d.date);
        })
        .y(function (d) {
            return y(d[biome]);
        });
};

class LineChart  {
    constructor(data, width, margin, height, colors, element) {

        this.data = data;
        this.width = width;
        this.margin = margin;
        this.height = height;
        this.colors = function(x) {
            var text = String(x);
            if (text.match(/^desert.*/i) != null) { return colors[0]; }
            else if (text.match(/^plains.*/i) != null) { return colors[1]; }
            else if (text.match(/^jungle.*/i) != null) { return colors[2]; }
            else if (text.match(/^wetlands.*/i) != null) { return colors[3]; }
            else {return 'Black';}
        };
        this.element = element;

        this.x = d3.scaleTime().range([0, this.width]);
        this.y = d3.scaleLinear().range([this.height, 0]);

        this.svg = d3.select(element).append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        this.ymax = d3.max(data, function(d) { return d3.max([d.Wetlands_Water, d.Jungle_Water, d.Plains_Water, d.Desert_Water]); });
        this.ymin = 0;

        this.x.domain(d3.extent(data, function(d) { return d.date; }));
        this.y.domain([this.ymin, this.ymax]);

        //add the X Axis
        this.svg.append("g")
            .attr("transform", "translate(0," + this.height + ")")
            .attr("class", "axis")
            .call(d3.axisBottom(this.x)
                .ticks(10)
                .tickFormat(format));

        // add the Y Axis
        this.svg.append("g")
            .call(d3.axisLeft(this.y).ticks(parseInt(height/25)));
    }

    lineStyle (biome) {
        var text = String(biome);
        if (text.match(/.*lv1.*/i) != null) { return 'lv1'; }
        else if (text.match(/.*lv2.*/i) != null) { return 'lv2'; }
        else if (text.match(/.*lv3.*/i) != null) { return 'lv3'; }
        else 'solid';
    }

    plotBiomeWater (data, biome, i) {
        var val_line = plot_line(biome, this.x, this.y);

        this.svg.append("path")
            .attr("class", "line")
            .attr("line_type", "biome_water_line")
            .attr("class", this.lineStyle(biome))
            .style('stroke', this.colors(biome))
            .style("stroke-width", 3)
            .attr("d", val_line(data));
    }

    set_ymax (BIOMES) {
        this.ymax = d3.max(this.data, function(d) { return 1.1*d3.max(BIOMES.map(function(biome) { return d[biome]; })) } );
    }

    plotBiomes (BIOMES, start=null, stop=null) {

        var data = null;
        if (start != null) {
            var data = this.data.filter(function (d) {
                if ((start - 30 >= 0) && (stop < this.data.slice(-1)[0].seconds)) {
                    return (d.seconds > start - 30 && d.seconds <= stop)
                } else if (start - 30 <= 0) {
                    return (d.seconds > 0 && d.seconds <= 100);
                } else {
                    return (d.seconds > this.data.slice(-1)[0].seconds - 100);
                }
            }, this);

            this.set_ymax(BIOMES);
            // this.ymax = d3.max(data, function(d) { return 1.1*d3.max(BIOMES.map(function(biome) { return d[biome]; })) } );
            this.ymin = 0;

            this.x.domain(d3.extent(data, function(d) { return d.date; }));
            this.y.domain([this.ymin, this.ymax]);

            this.svg.selectAll("g").remove();
            this.svg.selectAll("path").filter(function() {
                return !this.classList.contains('legend')
            }).remove();

            //add the X Axis
            this.svg.append("g")
                .attr("transform", "translate(0," + this.height + ")")
                .attr("class", "axis")
                .call(d3.axisBottom(this.x)
                    .ticks(10)
                    .tickFormat(format));

            // add the Y Axis
            this.svg.append("g")
                .call(d3.axisLeft(this.y).ticks(parseInt(height/25)));

        } else {
            data = this.data;
        }
        BIOMES.forEach(function(d, i) {
            var biome = BIOMES[i];
            this.plotBiomeWater(data, biome, i);
        }, this);
    }

    plotActiveRegions (start, moving, selected) {

        var time = start;
        var end_time = start;
        var important = false;
        var class_ = '';
        var raining = false;
        var override = false;

        while (time < this.data.length-1) {
            var finished = false;
            time += 1;

            selected.forEach(function(x) {
                if (this.data[start][regime_map[x]] != this.data[time][regime_map[x]]) {
                    if (this.data[start][x] >= 1) {
                        finished = true;
                        if (x.includes('Raining')) {
                            class_ = '_raining';
                        }
                        else if (x.includes('_up')) {
                            class_ = '_up';
                        }
                        else if (x.includes('_down')) {
                            class_ = '_down';
                        }
                    }
                }
                if (this.data[start][x] >= 1) {
                    important = true;
                }
                if (this.data[time][regime_map[x].replace('_regime', '_Raining')] >= 1){
                    if (selected.indexOf(regime_map[x].replace('_regime', '_Raining')) >= 0){
                        raining = true;
                    }
                }
            }, this);

            if (finished) {
                end_time = time;
                break;
            }
        }

        var area_start = start;
        var area_stop = end_time;

        if (end_time - start <= 10) {
            important = false;
        }

        var ixs = d3.range(area_start, area_stop).map(function (d) {
            var dte = new Date(2014, 4, 1);
            var u = +dte;
            var seconds = parseInt(d);
            var newU = u + seconds * 1000;
            var newD = new Date(newU);
            return newD
        });

        if (raining == true) {
            // important = false;
            class_ = '_raining'
        }

        if (important == true) {
            this.plotActiveRegion(ixs, moving, area_start, class_);
        }
        // else {
        //     var a_s = area_start == 0 ? 1 : area_start;
        //     this.plotActiveRegion(ixs, moving, a_s, '_not_important');
        // }

        return end_time;
    }

    plotFocus (start) {
        var area_start = start > 30 ? start - 30 : 0;
            area_start = area_start + 100 < this.data.length ? area_start : this.data.length - 100;
        var area_stop = area_start+100;

        var ixs = d3.range(area_start, area_stop).map(function (d) {
            var dte = new Date(2014, 4, 1);
            var u = +dte;
            var seconds = parseInt(d);
            var newU = u + seconds * 1000;
            var newD = new Date(newU);
            return newD
        });

        this.plotActiveRegion(ixs, true, area_start, '_view_window');
    }

    plotActiveRegion (x_indices, moving, id, class_='') {

        var area_tracker = this.svg.select('#area_' + parseInt(id))._groups[0][0]
        var x = this.x

        if (area_tracker == null) {

            if (moving) {
                this.svg.select('.area'+class_).remove()
            }

            // define the area
            var area = d3.area()
                .x(function (d) {
                    return x(d);
                })
                .y0(this.height)
                .y1(this.ymin);

            this.svg.append("path")
                .data([x_indices])
                .attr("class", "area"+class_)
                .attr("id", "area_" + parseInt(id))
                .attr("transform", null)
                .attr("d", area);

        }
    }

    plotLegend (biomes) {

        var legendRectHeight = 10;
        var legendRectWidth = 80;
        var legendSpacing = 65;
        var colors = this.colors;

        this.svg.selectAll('.legend').remove();

        biomes.forEach(function(d, i) {

            this.svg.append("text")
                    .attr("class", "legend")
                    .attr("id", "legend-text-" + i)
                    .attr("x", 0 + (legendSpacing * i) * 2)
                    .attr("y", (-(margin.top / 2)))
                    .style("fill", function () {
                        return colors(d)
                    })
                    .style("font-family", "sans-serif")
                    .style("font-size", "12px")
                    .text(d.replace('_', ' ').replace('Plains', 'Grasslands'));

            // adds line in the same color as used in the graph
            var thisItem = d3.select("#legend-text-" + i).node();
            var bb = thisItem.getBBox();
            var bx = bb.x - bb.width - legendRectWidth;
            this.svg.append("path")
                    .attr("class", "legend")
                    .attr("data-legend-key", i)
                    .attr("data-color", function () {
                        return colors(d)
                    })
                    .attr("d", "M" + (bb.x - 25) + "," + (bb.y + bb.height / 2) + " L" + (bb.x - 6) + "," + (bb.y + bb.height / 2))
                    .style("stroke", function () {
                        return colors(d)
                    })
                    .style("stroke-width", "2px")
                    .style("fill", "none")
                    .attr("class", this.lineStyle(d))
                    .attr("height", legendRectHeight)
                    .attr("width", legendRectWidth)
        }, this);
    }

    plotTimeTracker (seconds) {

        // we update the position of this line
        this.svg.select('.time_tracker').remove();

        var dte = new Date(2014, 4, 1);
        var u = +dte;
        var newU = u + seconds*1000;
        var newD = new Date(newU);

        var x = this.x

        this.svg.append("line")
            .attr("x1", x(newD))  //<<== change your code here
            .attr("y1", this.height)
            .attr("x2", x(newD))  //<<== and here
            .attr("y2", this.ymin)
            .attr('class', 'time_tracker')
            .attr("transform", null)
            .style("stroke-width", 3)
            .style("stroke", "black")
            .style("fill", "none");
    }

    plotInterval (start, stop) {
        alert('Hello World! -' + _args[0]);
    }

    attachTimeClickHandler (callback, charts) {
        var dte = new Date(2014, 4, 1);
        var offset = $(this.element).offset().left + this.margin.left;
        var x_axis = this.x;
        d3.select(this.element).on('click', function(){
            var selected_seconds = (x_axis.invert(d3.event.pageX - offset) - dte)/1000;
            callback(selected_seconds, charts);
        });
    }

};
