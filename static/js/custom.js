onscroll = function () {
    //your code here
    scrollTop = $(window).scrollTop();
    opacity = scrollTop / 300;
    blur = Math.min(2, scrollTop / 200);
    brightness = Math.max(0.5, 1 - scrollTop / 500);

    ['.latest','.articles'].forEach( function(e){
        $(e).css('opacity', opacity);
    })

    //$('body').css('backdrop-filter', `brightness(${brightness}) blur(${blur}px)`);
    $('body').css('backdrop-filter', `brightness(${brightness})`);
    $('body').css('background-position', `100% ${-0.15 * scrollTop}px`);
}

$(window).scroll(onscroll);

$(document).ready(onscroll); 