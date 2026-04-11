(function ($) {
  'use strict';

  // Hide the preloader once the DOM is ready so large images do not block first paint.
  $(function () {
    $('.preloader').fadeOut(100);
  });


  //  Search Form Open
  $('#searchOpen').on('click', function () {
    $('.search-wrapper').addClass('open');
  });
  $('#searchClose').on('click', function () {
    $('.search-wrapper').removeClass('open');
  });

  // featured post slider
  $('.featured-post-slider').slick({
    infinite: true,
    vertical: true,
    verticalSwiping: true,
    arrows: false,
    dots: true,
    responsive: [{
      breakpoint: 600,
      settings: {
        vertical: false,
        verticalSwiping: false,
      }
    }]
  });

  // venobox initialize
  $('.venobox').venobox();

})(jQuery);
