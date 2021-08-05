

function drawborder(pixels, color) {
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = color;
    ctx.lineWidth = pixels;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
}


function clearcells(color) {
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}


function drawcells(m, n, color, xs, ys) {
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");

    pixelwidth = canvas.width / n;
    pixelheight = canvas.height / m;

    console.log([pixelwidth, pixelheight]);

    ctx.fillStyle = color;
    xs.map(function(x, i) {
        const y = ys[i];
        ctx.fillRect(x*pixelwidth, y*pixelheight, pixelwidth, pixelheight);
    });
}

