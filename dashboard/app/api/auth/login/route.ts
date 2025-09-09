import { NextRequest, NextResponse } from 'next/server'
import { getUserById } from '@/lib/dataService'

export async function POST(request: NextRequest) {
  try {
    const { partnerId, userType } = await request.json()

    if (!partnerId || !userType) {
      return NextResponse.json(
        { error: 'Partner ID and user type are required' },
        { status: 400 }
      )
    }

    // Convert to number if it's a string
    const id = typeof partnerId === 'string' ? parseInt(partnerId) : partnerId

    // Find the user
    const user = await getUserById(id, userType as 'Driver' | 'Merchant')

    if (!user) {
      return NextResponse.json(
        { error: 'Partner not found' },
        { status: 404 }
      )
    }

    // Return user data (in real app, you'd validate credentials)
    return NextResponse.json({
      success: true,
      data: {
        partnerId: user.partner_id,
        partnerType: user.partner_type,
        creditScore: user.credit_score,
        rating: user.average_rating,
        tenure: user.tenure_months,
        location: user.location,
        demographics: {
          ageGroup: user.age_group,
          gender: user.gender,
          ethnicity: user.ethnicity,
          education: user.education_level
        },
        metrics: userType === 'Driver' ? {
          monthlyEarnings: user.monthly_earnings,
          completionRate: user.completion_rate,
          totalTrips: user.total_trips,
          vehicleType: user.vehicle_type,
          hoursOnlinePerWeek: user.hours_online_per_week
        } : {
          monthlySales: user.monthly_sales,
          orderAcceptanceRate: user.order_acceptance_rate,
          avgOrderValue: user.avg_order_value,
          merchantCategory: user.merchant_category,
          cuisineType: user.cuisine_or_store_type
        },
        fullData: user
      }
    })
  } catch (error) {
    console.error('Auth API Error:', error)
    return NextResponse.json(
      { error: 'Failed to authenticate partner' },
      { status: 500 }
    )
  }
}
