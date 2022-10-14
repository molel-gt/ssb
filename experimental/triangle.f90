! ------------------------------------------------------
! Compute the area of a triangle using Heron's formula
! ------------------------------------------------------

PROGRAM  HeronFormula
    IMPLICIT  NONE
 
    REAL     :: a, b, c             ! three sides
    REAL     :: s                   ! half of perimeter
    REAL     :: Area                ! triangle area
    LOGICAL  :: Cond_1, Cond_2      ! two logical conditions
 
    READ(*,*)  a, b, c
 
    WRITE(*,*)  "a = ", a
    WRITE(*,*)  "b = ", b
    WRITE(*,*)  "c = ", c
    WRITE(*,*)
 
    Cond_1 = (a > 0.) .AND. (b > 0.) .AND. (c > 0.0)
    Cond_2 = (a + b > c) .AND. (a + c > b) .AND. (b + c > a)
    IF (Cond_1 .AND. Cond_2) THEN
       s    = (a + b + c) / 2.0
       Area = SQRT(s * (s - a) * (s - b) * (s - c))
       WRITE(*,*) "Triangle area = ", Area
    ELSE
       WRITE(*,*) "ERROR: this is not a triangle!"
    END IF
 
 END PROGRAM  HeronFormula

program plotfunction
   implicit none
   integer :: i
   real :: x
   real, parameter :: xmin = 0.,xmax=10., a=-2.
   open(10, file='myplot.dat')
      do i = 1,100
         x = xmin + xmax*(i-1.0)/(100.0-1.0)
         write(10,*) x, f(x)
      enddo
   close(10)
end program plotfunction

function f(x)
   implicit none
   real :: f,x
   f = cos(x+a)
end function f

program addmats
   implicit none
   integer, parameter :: dimmat = 3
   real, dimension(dimmat, dimmat) :: a, b, c
   integer :: i, j
   a(1, 2) = 2.0
   do i=2, dimmat-1
      a(i, i+1) = 2.0
      b(i, i-1) = 1.0
   enddo
   b(dimmat, dimmat-1) = 1.0
   do i=1, dimmat
      do j=1, dimmat
         c(i, j) = a(i, j) + b(i, j)
      enddo
   enddo
   write(*,*) c
end program addmats