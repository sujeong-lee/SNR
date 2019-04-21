subroutine sici ( x, si )

!*****************************************************************************80
!
!! CISIB computes cosine and sine integrals.
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
!    they give permission to incorporate this routine into a user program
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    20 March 2012
!
!  Author:
!
!    Shanjie Zhang, Jianming Jin
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of Ci(x) and Si(x).
!
!    Output, real ( kind = 8 ) CI, SI, the values of Ci(x) and Si(x).
!
implicit none
!double precision, intent(out) :: ci
double precision fx
double precision gx
double precision, intent(out) ::  si
double precision, intent(in) ::  x
double precision x2

x2 = x * x

if ( x == 0.0D+00 ) then

!ci = -1.0D+300
si = 0.0D+00

else if ( x <= 1.0D+00 ) then

!ci = (((( -3.0D-08        * x2 &
!+ 3.10D-06     ) * x2 &
!- 2.3148D-04   ) * x2 &
!+ 1.041667D-02 ) * x2 &
!- 0.25D+00     ) * x2 + 0.577215665D+00 + log ( x )

si = (((( 3.1D-07        * x2 &
- 2.834D-05    ) * x2 &
+ 1.66667D-03  ) * x2 &
- 5.555556D-02 ) * x2 + 1.0D+00 ) * x

else

fx = (((( x2              &
+ 38.027264D+00  ) * x2 &
+ 265.187033D+00 ) * x2 &
+ 335.67732D+00  ) * x2 &
+ 38.102495D+00  ) /    &
(((( x2                 &
+ 40.021433D+00  ) * x2 &
+ 322.624911D+00 ) * x2 &
+ 570.23628D+00  ) * x2 &
+ 157.105423D+00 )

gx = (((( x2               &
+ 42.242855D+00  ) * x2  &
+ 302.757865D+00 ) * x2  &
+ 352.018498D+00 ) * x2  &
+ 21.821899D+00 ) /      &
(((( x2                  &
+ 48.196927D+00   ) * x2 &
+ 482.485984D+00  ) * x2 &
+ 1114.978885D+00 ) * x2 &
+ 449.690326D+00  ) / x

!ci = fx * sin ( x ) / x - gx * cos ( x ) / x

si = 1.570796327D+00 - fx * cos ( x ) / x - gx * sin ( x ) / x

end if

return
end


subroutine eval_legendre(n,x,pl)
!======================================
! calculates Legendre polynomials Pn(x)
! using the recurrence relation
! if n > 100 the function retuns 0.0
!======================================
double precision, intent(out) ::  pl
double precision, intent(in) :: x
double precision pln(0:n)
integer, intent(in) :: n
integer k

pln(0) = 1.0
pln(1) = x

if (n <= 1) then
pl = pln(n)
else
do k=1,n-1
pln(k+1) = ((2.0*k+1.0)*x*pln(k) - float(k)*pln(k-1))/(float(k+1))
end do
pl = pln(n)
end if
return
end

